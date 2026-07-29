# =============================================================================
# STANDALONE NOTEBOOK — Dense Autoencoder on Spirit (Fully Independent)
#
# Based on [Bekkouche2025_BiLSTM] — F1-sensitive threshold search dramatically
#   outperforms the classic mean+2σ heuristic for reconstruction-error thresholds.
# Based on [Bekkouche2024] — AE+Clustering framework; unsupervised AE baseline
#   trained on normal samples only to learn a compact latent representation.
#   It uses K-Means to cluster latent representations (reduced to 4D via PCA),
#   and flags anomalies using a logical OR of reconstruction error and clustering distance.
#
# ✅ ZERO dependencies — reads raw Spirit_Drain.csv directly.
# ✅ Chunked reading for large Spirit file → safe on Kaggle memory limits.
# ✅ Builds its own TF-IDF splits inline then frees the raw data.
# ✅ One dataset only → RAM stays safe on Kaggle (T4/P100).
# ✅ Checkpoint system → re-running skips completed steps.
#
# Kaggle setup:
#   - Add dataset: pfe-log-anomaly  (contains Spirit_Drain.csv)
#   - Accelerator: GPU T4 x2 or P100
#   - Estimated time: ~20 minutes
# =============================================================================

import os, gc, json, pathlib, time, random, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import torch
import torch.nn as nn
import scipy.sparse
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, matthews_corrcoef,
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Environment & Paths
# ─────────────────────────────────────────────────────────────────────────────
KAGGLE   = os.path.exists('/kaggle')
BASE_IN  = '/kaggle/input/pfe-log-anomaly' if KAGGLE else 'Dataset'
# NOTE: This result folder does not yet exist on disk. The model was implemented
# but evaluation under the primary temporal-split protocol is pending.
BASE_OUT = '/kaggle/working'               if KAGGLE else 'result/results_DenseAE_Spirit'
REPORT   = f'{BASE_OUT}/pfe_report'
os.makedirs(f'{BASE_OUT}/models', exist_ok=True)
os.makedirs(REPORT, exist_ok=True)

DS_KEY      = 'spirit'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Safety cap: set to an integer (e.g. 5_000_000) if RAM is tight; None = full file
NROWS_LIMIT = None
CHUNK_SIZE  = 500_000    # rows per chunk during chunked CSV reading
print(f"Device: {DEVICE}")

CKPT = pathlib.Path(BASE_OUT) / f'ckpt_ae_{DS_KEY}.json'
def save_ckpt(d):
    with open(CKPT, 'w') as f: json.dump(d, f)
def load_ckpt():
    if CKPT.exists():
        with open(CKPT) as f: return json.load(f)
    return {}
ckpt = load_ckpt()

def find_file(name):
    name_lower = name.lower()
    search_dir = '/kaggle/input' if os.path.exists('/kaggle') else '.'
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.lower() == name_lower:
                return os.path.join(root, f)
    # If not found, list what we did find to help debugging
    all_files = []
    for root, _, files in os.walk(search_dir):
        for f in files:
            all_files.append(os.path.join(root, f))
    files_str = "\n".join(all_files[:15])
    if len(all_files) > 15:
        files_str += f"\n... and {len(all_files)-15} more files."
    raise FileNotFoundError(
        f"'{name}' not found under {search_dir}.\n"
        f"Available files in search path:\n{files_str}"
    )
print(f"{'Kaggle' if KAGGLE else 'Local'} | Dense AE on Spirit — Standalone")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Load Spirit + Build TF-IDF Splits Inline  (Chunked Reading)
# ─────────────────────────────────────────────────────────────────────────────
TFIDF_PARAMS = dict(
    max_features=10_000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=2,
    token_pattern=r'[a-zA-Z_:\-\.]+',
)

class SparseDataset(torch.utils.data.Dataset):
    def __init__(self, X_sparse):
        self.X_sparse = X_sparse
    def __len__(self):
        return self.X_sparse.shape[0]
    def __getitem__(self, idx):
        return idx

def make_collate_fn(X_sparse):
    def collate_fn(batch_indices):
        dense_batch = X_sparse[batch_indices].toarray().astype(np.float32)
        return torch.from_numpy(dense_batch)
    return collate_fn

if 'data_ready' not in ckpt:
    print("\n[CELL 2] Loading Spirit_Drain.csv in chunks ...")
    t0 = time.time()

    filepath = find_file('Spirit_Drain.csv')

    # Chunked read to collect templates & labels
    templates_list, labels_list = [], []
    total_rows = 0
    reader = pd.read_csv(
        filepath,
        usecols=['template', 'label'],
        on_bad_lines='skip',
        low_memory=False,
        chunksize=CHUNK_SIZE,
    )
    for chunk in reader:
        templates_list.append(chunk['template'].fillna('').values)
        # Spirit label: '-' = normal, anything else = anomaly [Bekkouche2025_Spirit]
        raw_label = chunk['label'].astype(str).str.strip()
        labels_list.append((raw_label != '-').astype(np.int8).values)
        total_rows += len(chunk)
        if NROWS_LIMIT is not None and total_rows >= NROWS_LIMIT:
            print(f"  ⚠️  NROWS_LIMIT={NROWS_LIMIT:,} reached — truncating.")
            break
        del chunk; gc.collect()

    templates = np.concatenate(templates_list).astype(object)
    labels    = np.concatenate(labels_list).astype(np.int8)
    del templates_list, labels_list; gc.collect()
    print(f"  Loaded: {len(templates):,} rows")
    print(f"  Normal: {(labels==0).sum():,} | Anomaly: {(labels==1).sum():,} "
          f"({labels.mean()*100:.1f}%)")

    # Stratified random split 70/10/20 (80/20 trainval vs test)
    from sklearn.model_selection import train_test_split
    n = len(labels)
    indices = np.arange(n)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])
    y_train = labels[train_idx].astype(np.int32)
    y_val   = labels[val_idx].astype(np.int32)
    y_test  = labels[test_idx].astype(np.int32)
    print(f"  Split → train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}")

    # TF-IDF (fit on TRAIN only)
    print("  Building TF-IDF features ...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_sp = vectorizer.fit_transform(templates[train_idx])
    X_val_sp   = vectorizer.transform(templates[val_idx])
    X_test_sp  = vectorizer.transform(templates[test_idx])
    INPUT_DIM  = X_train_sp.shape[1]
    print(f"  Vocab: {INPUT_DIM} | Shapes: tr={X_train_sp.shape} v={X_val_sp.shape} te={X_test_sp.shape}")

    # Save sparse splits for checkpoint recovery
    scipy.sparse.save_npz(f'{BASE_OUT}/models/ae_X_train_{DS_KEY}.npz', X_train_sp)
    scipy.sparse.save_npz(f'{BASE_OUT}/models/ae_X_val_{DS_KEY}.npz', X_val_sp)
    scipy.sparse.save_npz(f'{BASE_OUT}/models/ae_X_test_{DS_KEY}.npz', X_test_sp)
    np.savez_compressed(f'{BASE_OUT}/models/ae_y_{DS_KEY}.npz', y_train=y_train, y_val=y_val, y_test=y_test)
    del templates, labels; gc.collect()
    print(f"  ✅ TF-IDF done ({time.time()-t0:.0f}s)")

    ckpt['data_ready'] = True
    ckpt['input_dim']  = int(INPUT_DIM)
    save_ckpt(ckpt)

else:
    print("[CELL 2] ⏭️  Checkpoint found — loading splits ...")
    X_train_sp = scipy.sparse.load_npz(f'{BASE_OUT}/models/ae_X_train_{DS_KEY}.npz')
    X_val_sp   = scipy.sparse.load_npz(f'{BASE_OUT}/models/ae_X_val_{DS_KEY}.npz')
    X_test_sp  = scipy.sparse.load_npz(f'{BASE_OUT}/models/ae_X_test_{DS_KEY}.npz')
    y_splits   = np.load(f'{BASE_OUT}/models/ae_y_{DS_KEY}.npz')
    y_train    = y_splits['y_train']
    y_val      = y_splits['y_val']
    y_test     = y_splits['y_test']
    INPUT_DIM  = ckpt.get('input_dim', X_train_sp.shape[1])

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Dense Autoencoder Architecture
# ─────────────────────────────────────────────────────────────────────────────
class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck, n_hidden, dropout):
        super().__init__()
        enc_layers = []
        in_ch = input_dim
        for _ in range(n_hidden):
            enc_layers += [nn.Linear(in_ch, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_ch = hidden_dim
        enc_layers += [nn.Linear(in_ch, bottleneck), nn.ReLU()]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        in_ch = bottleneck
        for _ in range(n_hidden):
            dec_layers += [nn.Linear(in_ch, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_ch = hidden_dim
        dec_layers += [nn.Linear(in_ch, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(model, X_normal_train, X_normal_val,
                      lr, batch_size, max_epochs=100, patience=15):
    model.to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimiser, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    t_dataset = SparseDataset(X_normal_train)
    loader   = DataLoader(t_dataset, batch_size=batch_size,
                          shuffle=True, collate_fn=make_collate_fn(X_normal_train),
                          pin_memory=(DEVICE.type == 'cuda'))

    best_val_loss = float('inf')
    best_state    = None
    no_improve    = 0
    train_losses, val_losses = [], []

    for epoch in range(max_epochs):
        model.train()
        ep_loss = 0.0
        for xb in loader:
            xb = xb.to(DEVICE)
            optimiser.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            optimiser.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(t_dataset)

        model.eval()
        val_loss = 0.0
        val_dataset = SparseDataset(X_normal_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, collate_fn=make_collate_fn(X_normal_val))
        with torch.no_grad():
            for xb in val_loader:
                xb = xb.to(DEVICE)
                loss = criterion(model(xb), xb)
                val_loss += loss.item() * len(xb)
        vl = val_loss / len(val_dataset)
        scheduler.step(vl)

        train_losses.append(ep_loss)
        val_losses.append(vl)

        if vl < best_val_loss - 1e-6:
            best_val_loss = vl
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return train_losses, val_losses, best_val_loss


def compute_errors(model, X_sparse, batch_size=1024):
    model.eval()
    errors = []
    n_samples = X_sparse.shape[0]
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            xb_np = X_sparse[i:i+batch_size].toarray().astype(np.float32)
            xb = torch.from_numpy(xb_np).to(DEVICE)
            rec = model(xb)
            mse = ((rec - xb) ** 2).mean(dim=1).cpu().numpy()
            errors.append(mse)
    return np.concatenate(errors).astype(np.float32)


def get_latent_representations(model, X_sparse, batch_size=1024):
    model.eval()
    latents = []
    n_samples = X_sparse.shape[0]
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            xb_np = X_sparse[i:i+batch_size].toarray().astype(np.float32)
            xb = torch.from_numpy(xb_np).to(DEVICE)
            z = model.encoder(xb).cpu().numpy()
            latents.append(z)
    return np.concatenate(latents).astype(np.float32)


def ae_clustering_threshold_search(errors_all, dists_all, threshold_d, y_true, n_points=1000):
    lo = float(errors_all.min())
    hi = float(np.percentile(errors_all, 99.5))
    thresholds = np.linspace(lo, hi, n_points)
    best_thr_e, best_f1 = lo, 0.0
    
    anom_d = (dists_all >= threshold_d)
    for thr in thresholds:
        anom_e = (errors_all >= thr)
        preds = (anom_e | anom_d).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr_e = thr
    return float(best_thr_e), float(best_f1)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Optuna Hyper-parameter Search (15 trials)
# ─────────────────────────────────────────────────────────────────────────────
mask_train_normal = (y_train == 0)
mask_val_normal   = (y_val   == 0)
X_tr_norm = X_train_sp[mask_train_normal]
X_vl_norm = X_val_sp[mask_val_normal]
print(f"\nNormal train: {X_tr_norm.shape[0]:,} | Normal val: {X_vl_norm.shape[0]:,}")

if 'ae_done' not in ckpt:
    print("\n[CELL 4] Optuna hyper-parameter search (15 trials) ...")
    t0 = time.time()

    def ae_objective(trial):
        hidden_dim = trial.suggest_categorical('hidden_dim',  [128, 256, 512])
        bottleneck = trial.suggest_categorical('bottleneck',  [16, 32, 64])
        n_hidden   = trial.suggest_int('n_hidden', 2, 4)
        dropout    = trial.suggest_float('dropout', 0.1, 0.4)
        lr         = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])

        model = DenseAutoencoder(INPUT_DIM, hidden_dim, bottleneck, n_hidden, dropout)
        _, _, _ = train_autoencoder(
            model, X_tr_norm, X_vl_norm,
            lr=lr, batch_size=batch_size, max_epochs=20, patience=5
        )
        val_errors = compute_errors(model, X_val_sp)
        
        # Fit K-Means on latent representations (projected to 4D) [Bekkouche2024]
        latents_tr = get_latent_representations(model, X_tr_norm)
        pca = PCA(n_components=4, random_state=42)
        latents_tr_4d = pca.fit_transform(latents_tr)
        
        kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=128, n_init='auto')
        kmeans.fit(latents_tr_4d)
        
        # Validation distance
        latents_vl = get_latent_representations(model, X_val_sp)
        latents_vl_4d = pca.transform(latents_vl)
        val_dists = np.min(kmeans.transform(latents_vl_4d), axis=1)
        
        # Normal validation distance threshold (95th percentile)
        latents_vl_norm = get_latent_representations(model, X_vl_norm)
        latents_vl_norm_4d = pca.transform(latents_vl_norm)
        val_norm_dists = np.min(kmeans.transform(latents_vl_norm_4d), axis=1)
        thr_d = np.percentile(val_norm_dists, 95)
        
        _, val_f1 = ae_clustering_threshold_search(val_errors, val_dists, thr_d, y_val)
        
        del model, pca, kmeans; gc.collect()
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        return val_f1

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    study.enqueue_trial({
        'hidden_dim': 256, 'bottleneck': 32, 'n_hidden': 2,
        'dropout': 0.2, 'lr': 0.001, 'batch_size': 256
    })
    study.optimize(ae_objective, n_trials=15, timeout=900)
    best_params = study.best_params
    print(f"  🏆 Best params: {best_params} → Val F1={study.best_value:.4f}")

    # ── Full Training with Best Params ───────────────────────────────────────
    print("\n[CELL 4b] Full training ...")
    t1 = time.time()
    ae = DenseAutoencoder(
        INPUT_DIM,
        hidden_dim = best_params['hidden_dim'],
        bottleneck = best_params['bottleneck'],
        n_hidden   = best_params['n_hidden'],
        dropout    = best_params['dropout'],
    )
    train_losses, val_losses, _ = train_autoencoder(
        ae, X_tr_norm, X_vl_norm,
        lr         = best_params['lr'],
        batch_size = best_params['batch_size'],
        max_epochs = 100,
        patience   = 15,
    )
    print(f"  Training done in {time.time()-t1:.0f}s")

    # Fit final K-Means and PCA [Bekkouche2024]
    print("  Fitting PCA + MiniBatchKMeans on latent representations ...")
    latents_train = get_latent_representations(ae, X_tr_norm)
    pca = PCA(n_components=4, random_state=42)
    latents_train_4d = pca.fit_transform(latents_train)
    
    kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=256, n_init='auto')
    kmeans.fit(latents_train_4d)

    # ── F1-Optimal Threshold on Validation Set ───────────────────────────────
    val_errors = compute_errors(ae, X_val_sp)
    latents_val = get_latent_representations(ae, X_val_sp)
    latents_val_4d = pca.transform(latents_val)
    val_dists = np.min(kmeans.transform(latents_val_4d), axis=1)

    latents_val_norm = get_latent_representations(ae, X_vl_norm)
    latents_val_norm_4d = pca.transform(latents_val_norm)
    val_norm_dists = np.min(kmeans.transform(latents_val_norm_4d), axis=1)
    threshold_d = float(np.percentile(val_norm_dists, 95))

    threshold_e, val_f1 = ae_clustering_threshold_search(val_errors, val_dists, threshold_d, y_val)
    print(f"  📐 Optimal Reconstruction threshold: {threshold_e:.6f}")
    print(f"  📐 Optimal Clustering threshold (95th percentile): {threshold_d:.6f} (Val F1={val_f1:.4f})")

    # ── Test Evaluation ───────────────────────────────────────────────────────
    print("  Evaluating on test set ...")
    t_inf = time.time()
    test_errors = compute_errors(ae, X_test_sp)
    latents_test = get_latent_representations(ae, X_test_sp)
    latents_test_4d = pca.transform(latents_test)
    test_dists = np.min(kmeans.transform(latents_test_4d), axis=1)
    infer_time  = time.time() - t_inf

    # Dual Flagging logical OR
    y_pred = ((test_errors >= threshold_e) | (test_dists >= threshold_d)).astype(int)

    try:
        auc_score = roc_auc_score(y_test, test_errors)
    except Exception:
        auc_score = float('nan')

    metrics = {
        'Dataset':    'Spirit',
        'Model':      'DenseAE_Clustering',
        'Type':       'Unsupervised (AE)',
        'Precision':  round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'Recall':     round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'F1_Anomaly': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'Macro_F1':   round(float(f1_score(y_test, y_pred, average='macro', zero_division=0)), 4),
        'AUC':        round(float(auc_score), 4),
        'MCC':        round(float(matthews_corrcoef(y_test, y_pred)), 4),
        'Threshold_Reconstruction': round(float(threshold_e), 6),
        'Threshold_Clustering':     round(float(threshold_d), 6),
        'Inference_Time_s':        round(infer_time, 4),
        'Inference_Per_Sample_ms': round(infer_time / len(y_test) * 1000, 4),
    }

    print(f"\n  📊 TEST RESULTS — Spirit AE + Clustering:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly'], digits=4))

    # Save
    torch.save(ae.state_dict(), f'{BASE_OUT}/models/ae_{DS_KEY}_opt.pt')
    joblib.dump(pca, f'{BASE_OUT}/models/ae_{DS_KEY}_pca.pkl')
    joblib.dump(kmeans, f'{BASE_OUT}/models/ae_{DS_KEY}_kmeans.pkl')

    with open(f'{BASE_OUT}/models/ae_{DS_KEY}_threshold.json', 'w') as f:
        json.dump({'threshold_e': threshold_e, 'threshold_d': threshold_d, **best_params, **metrics}, f, indent=2)
    pd.DataFrame([metrics]).to_csv(f'{REPORT}/ae_spirit_results.csv', index=False)

    # Loss curves
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label='Train MSE', color='steelblue')
    ax.plot(val_losses,   label='Val MSE',   color='darkorange')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.set_title('Dense AE — Loss Curve (Spirit)', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f'{REPORT}/ae_loss_spirit.png', dpi=300); plt.close()

    # Reconstruction Error Histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    norm_err = test_errors[y_test == 0]
    anom_err = test_errors[y_test == 1]
    ax.hist(norm_err, bins=80, alpha=0.6, color='green',  label='Normal',  density=True)
    ax.hist(anom_err, bins=80, alpha=0.6, color='red',    label='Anomaly', density=True)
    ax.axvline(threshold_e, color='black', linestyle='--', lw=2, label=f'Threshold={threshold_e:.4f}')
    ax.set_xlabel('Reconstruction Error (MSE)')
    ax.set_ylabel('Density')
    ax.set_title('Dense AE — Reconstruction Error Distribution (Spirit)', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f'{REPORT}/ae_hist_spirit.png', dpi=300); plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    ax.set_title('Dense AE + Clustering CM — Spirit', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{REPORT}/ae_cm_spirit.png', dpi=300); plt.close()

    del ae, val_errors, test_errors, val_norm_dists, pca, kmeans; gc.collect()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()

    ckpt['ae_done'] = True
    ckpt['best_params'] = best_params
    ckpt['threshold_e'] = float(threshold_e)
    ckpt['threshold_d'] = float(threshold_d)
    save_ckpt(ckpt)
    print(f"\n  ✅ Dense AE + Clustering Spirit done in {time.time()-t0:.0f}s")

else:
    print("[CELL 4] ⏭️  AE already done (checkpoint found)")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Final Memory Cleanup
# ─────────────────────────────────────────────────────────────────────────────
try:
    del X_train_sp, X_val_sp, X_test_sp
    del X_tr_norm, X_vl_norm
    del y_train, y_val, y_test
    gc.collect()
except NameError:
    pass
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Verification Block
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  ✅ Dense AE + Clustering Spirit STANDALONE — COMPLETE")
print(f"{'='*60}")
output_files = [
    (f'{BASE_OUT}/models/ae_{DS_KEY}_opt.pt',         'ae_spirit_opt.pt'),
    (f'{BASE_OUT}/models/ae_{DS_KEY}_pca.pkl',        'ae_spirit_pca.pkl'),
    (f'{BASE_OUT}/models/ae_{DS_KEY}_kmeans.pkl',     'ae_spirit_kmeans.pkl'),
    (f'{BASE_OUT}/models/ae_{DS_KEY}_threshold.json',  'ae_spirit_threshold.json'),
    (f'{REPORT}/ae_spirit_results.csv',                'ae_spirit_results.csv'),
    (f'{REPORT}/ae_loss_spirit.png',                   'ae_loss_spirit.png'),
    (f'{REPORT}/ae_hist_spirit.png',                   'ae_hist_spirit.png'),
    (f'{REPORT}/ae_cm_spirit.png',                     'ae_cm_spirit.png'),
]
for fpath, fname in output_files:
    exists = os.path.exists(fpath)
    print(f"  {'✅' if exists else '❌'} {fname}")

if ckpt.get('ae_done'):
    print(f"\n  📌 Best params: {ckpt.get('best_params', {})}")
    print(f"  📐 Threshold Reconstruction: {ckpt.get('threshold_e', 'N/A'):.6f}")
    print(f"  📐 Threshold Clustering:     {ckpt.get('threshold_d', 'N/A'):.6f}")
print(f"\n  📊 Results → {REPORT}/ae_spirit_results.csv")
print(f"{'='*60}")

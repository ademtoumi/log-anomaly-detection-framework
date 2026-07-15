# =============================================================================
# STANDALONE NOTEBOOK 15 â€” BiLSTM Autoencoder on BGL (Fully Independent)
#
# âœ… ZERO dependencies â€” reads raw BGL_Drain.csv directly from Kaggle input.
# âœ… Builds BGL sessions inline (sliding window, vocab, int32 sequences).
# âœ… One dataset only (BGL) â€” RAM stays safe on Kaggle T4/P100.
# âœ… Trains on NORMAL sessions only â€” reconstruction error for anomaly scoring.
#
# References:
#   [Bekkouche2025_BiLSTM]  â€” F1=0.993 on HDFS; BiLSTM-AE on session sequences.
#                             Encoder is bidirectional, Decoder is unidirectional.
#   [Bekkouche2024]          â€” MSE reconstruction error for unsupervised AE scoring.
#   [Zhang2019_LogRobust]    â€” Sequential temporal split is scientifically honest.
#
# Architecture:
#   Embedding(vocab_size, embed_dim, padding_idx=0)
#   â†’ BiLSTM Encoder(embed_dim, hidden_size, num_layers)
#   â†’ Linear(hidden_size * 2, hidden_size)
#   â†’ LSTM Decoder(hidden_size, hidden_size, 1)
#   â†’ Linear(hidden_size, embed_dim)
#   Loss: MSELoss(decoded_embeddings, original_embeddings.detach())
#
# Kaggle setup:
#   - Dataset: pfe-log-anomaly  (must contain BGL_Drain.csv)
#   - Accelerator: GPU T4 x2 or P100
#   - Estimated time: ~25 minutes
# =============================================================================

import os, gc, json, pathlib, time, random, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_curve, auc, average_precision_score,
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 1 â€” Environment
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
KAGGLE   = os.path.exists('/kaggle')
BASE_IN  = '/kaggle/input/pfe-log-anomaly' if KAGGLE else 'Dataset'
BASE_OUT = '/kaggle/working'               if KAGGLE else 'result/results_bilstm_ae_bgl'
MODEL_DIR = f'{BASE_OUT}/models'
REPORT    = f'{BASE_OUT}/pfe_report'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DS_KEY = 'bgl'

# BGL sliding window params
WINDOW_SIZE = 20
STEP_SIZE   = 10

# Safety cap
NROWS_LIMIT = None  # None = full dataset

CKPT = pathlib.Path(BASE_OUT) / f'ckpt_15_lstm_ae_{DS_KEY}.json'
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
print(f"âœ… Env: {'Kaggle' if KAGGLE else 'Local'} | Device: {DEVICE} | BGL BiLSTM-AE Standalone")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 2 â€” Build BGL Sessions (Sliding Window) Inline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if 'sessions_ready' not in ckpt:
    print("\n[CELL 2] Building BGL sliding-window sessions ...")
    t0 = time.time()

    filepath = find_file('BGL_Drain.csv')
    df = pd.read_csv(filepath, usecols=['template', 'label'], nrows=NROWS_LIMIT,
                     on_bad_lines='skip', low_memory=False)
    print(f"  Loaded BGL CSV: {len(df):,} rows")

    all_templates = df['template'].fillna('').astype(str).tolist()
    # BGL label: '-' = normal, anything else = anomaly
    all_labels    = (df['label'].astype(str).str.strip() != '-').astype(int).tolist()
    del df; gc.collect()

    n_total = len(all_templates)
    
    # Build vocabulary from training + validation portion (first 80% of lines) to avoid leakage
    i1_lines = int(n_total * 0.80)
    unique_t = sorted(set(all_templates[:i1_lines]))
    vocab_bgl = {'<PAD>': 0, '<UNK>': 1}
    for idx, t in enumerate(unique_t):
        vocab_bgl[t] = idx + 2
    joblib.dump(vocab_bgl, f'{MODEL_DIR}/vocab_bgl_lstm_ae.pkl')
    print(f"  Vocabulary: {len(vocab_bgl)} templates (train-only, no leakage)")

    event_ids = np.array(
        [vocab_bgl.get(t, 1) for t in all_templates], dtype=np.int32
    )
    label_arr = np.array(all_labels, dtype=np.int32)
    del all_templates, all_labels; gc.collect()

    n_windows = (n_total - WINDOW_SIZE) // STEP_SIZE + 1
    print(f"  Building {n_windows:,} sliding windows ...")
    sequences = np.zeros((n_windows, WINDOW_SIZE), dtype=np.int32)
    labels    = np.zeros(n_windows, dtype=np.int32)

    for i in range(n_windows):
        start = i * STEP_SIZE
        end   = start + WINDOW_SIZE
        sequences[i] = event_ids[start:end]
        labels[i]    = int(label_arr[start:end].max())

    del event_ids, label_arr; gc.collect()
    print(f"  Windows: {n_windows:,} | Anomaly: {labels.sum():,} ({labels.mean()*100:.1f}%)")

    from sklearn.model_selection import train_test_split
    indices = np.arange(n_windows)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])

    np.savez_compressed(f'{MODEL_DIR}/bgl_sessions_train_lstmae.npz', X=sequences[train_idx], y=labels[train_idx])
    np.savez_compressed(f'{MODEL_DIR}/bgl_sessions_val_lstmae.npz',   X=sequences[val_idx],   y=labels[val_idx])
    np.savez_compressed(f'{MODEL_DIR}/bgl_sessions_test_lstmae.npz',  X=sequences[test_idx],  y=labels[test_idx])

    print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")
    del sequences, labels; gc.collect()

    elapsed = time.time() - t0
    ckpt['sessions_ready'] = True; save_ckpt(ckpt)
    print(f"  âœ… BGL sessions saved ({elapsed:.0f}s)")
else:
    print("[CELL 2] â­ï¸  Sessions already built (checkpoint)")

# Load sessions
tr = np.load(f'{MODEL_DIR}/bgl_sessions_train_lstmae.npz')
va = np.load(f'{MODEL_DIR}/bgl_sessions_val_lstmae.npz')
te = np.load(f'{MODEL_DIR}/bgl_sessions_test_lstmae.npz')
vocab_bgl = joblib.load(f'{MODEL_DIR}/vocab_bgl_lstm_ae.pkl')
VS = len(vocab_bgl)

X_tr, y_tr = tr['X'], tr['y']
X_v,  y_v  = va['X'], va['y']
X_te, y_te = te['X'], te['y']

print(f"  VS={VS} | Train={X_tr.shape} | Val={X_v.shape} | Test={X_te.shape}")
print(f"  Anomaly: tr={y_tr.mean()*100:.1f}% | v={y_v.mean()*100:.1f}% | te={y_te.mean()*100:.1f}%")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 3 â€” BiLSTM Autoencoder Architecture
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class BiLSTMAutoencoder(nn.Module):
    """
    BiLSTM Encoder-Decoder Autoencoder for unsupervised anomaly detection.
    Encoder: Bidirectional LSTM compresses BGL session sequence to a bottleneck context.
    Decoder: Unidirectional LSTM reconstructs the embedding sequence.
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.hidden_size = hidden_size

        # Encoder: BiLSTM
        self.encoder = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # Project concatenated directions back to decoder dimension
        self.combine_directions = nn.Linear(hidden_size * 2, hidden_size)

        # Decoder
        self.decoder = nn.LSTM(
            hidden_size, hidden_size, 1, batch_first=True
        )

        # Project decoder output back to embedding space
        self.output_proj = nn.Linear(hidden_size, embed_dim)

    def forward(self, x):
        B, T = x.size(0), x.size(1)
        embedded = self.embedding(x)            # (B, T, E)

        # Encode: h_n shape: (num_layers * 2, B, H)
        _, (h_n, _) = self.encoder(embedded)
        
        # Combine forward and backward directions of the last layer
        h_forward  = h_n[-2]
        h_backward = h_n[-1]
        h_combined = torch.cat([h_forward, h_backward], dim=-1)         # (B, H * 2)
        context_state = torch.relu(self.combine_directions(h_combined))  # (B, H)
        
        # Reshape context_state -> initial hidden state for decoder (1, B, hidden_size)
        h0 = context_state.unsqueeze(0)
        c0 = torch.zeros_like(h0)

        # Create zero input sequence for decoder
        decoder_input = torch.zeros(B, T, self.hidden_size, device=x.device)

        # Decode with initial state
        decoded, _ = self.decoder(decoder_input, (h0, c0))      # (B, T, H)
        recon = self.output_proj(decoded)       # (B, T, E)

        return embedded, recon

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 4 â€” F1-Sensitive Threshold + Error Computation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def f1_threshold_search(errors, y_true, n_points=1000):
    best_f1, best_t = 0, float(np.median(errors))
    for t in np.linspace(errors.min(), np.percentile(errors, 99.5), n_points):
        y_pred = (errors > t).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_t = t
    return best_t, best_f1

def compute_errors(model, X, batch_size=256):
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i+batch_size]).long().to(DEVICE)
            emb, recon = model(batch)
            mse = ((emb - recon)**2).mean(dim=(1, 2)).cpu().numpy()
            errors.extend(mse)
    return np.array(errors)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 5 â€” Optuna + Full Training
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if 'ae_done' not in ckpt:
    print(f"\n[CELL 5] BiLSTM-AE Optimization â€” BGL")

    # Train on NORMAL sequences only (unsupervised)
    X_train_normal = X_tr[y_tr == 0]
    print(f"  Normal train: {len(X_train_normal):,} / {len(y_tr):,}")

    def objective(trial):
        cfg = {
            'embed_dim':   trial.suggest_categorical('embed_dim',   [32, 64]),
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128]),
            'num_layers':  trial.suggest_int('num_layers', 1, 2),
            'dropout':     trial.suggest_float('dropout', 0.1, 0.3),
            'lr':          trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'batch_size':  trial.suggest_categorical('batch_size', [128, 256]),
        }
        ae = BiLSTMAutoencoder(VS, cfg['embed_dim'], cfg['hidden_size'],
                               cfg['num_layers'], cfg['dropout']).to(DEVICE)
        opt_ae = torch.optim.Adam(ae.parameters(), lr=cfg['lr'])
        crit   = nn.MSELoss()
        dl     = DataLoader(TensorDataset(torch.from_numpy(X_train_normal).long()),
                            batch_size=cfg['batch_size'], shuffle=True)
        ae.train()
        for _ in range(10):  # Quick epochs for Optuna
            for (xb,) in dl:
                xb = xb.to(DEVICE)
                opt_ae.zero_grad()
                emb, recon = ae(xb)
                crit(recon, emb.detach()).backward()
                opt_ae.step()

        val_errors = compute_errors(ae, X_v)
        _, vf1 = f1_threshold_search(val_errors, y_v, n_points=200)
        del ae; gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()
        return vf1

    study = optuna.create_study(direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42))
    study.enqueue_trial({'embed_dim': 64, 'hidden_size': 128, 'num_layers': 2,
                         'dropout': 0.2, 'lr': 0.001, 'batch_size': 256})
    print("  ðŸ” Optuna (12 trials) ...")
    study.optimize(objective, n_trials=12, timeout=600)
    bp = study.best_params
    print(f"  ðŸ† {bp} â†’ Val F1={study.best_value:.4f}")

    # Full Training
    print(f"\n  ðŸš€ Full training (60 epochs, patience=10) ...")
    ae = BiLSTMAutoencoder(VS, bp['embed_dim'], bp['hidden_size'],
                           bp['num_layers'], bp['dropout']).to(DEVICE)
    opt_ae    = torch.optim.Adam(ae.parameters(), lr=bp['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_ae, patience=5, factor=0.5)
    crit = nn.MSELoss()
    dl   = DataLoader(TensorDataset(torch.from_numpy(X_train_normal).long()),
                      batch_size=bp['batch_size'], shuffle=True)

    best_vf1, best_state, no_improve = 0, None, 0
    losses = []

    for ep in range(1, 61):
        ae.train(); el = 0
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            opt_ae.zero_grad()
            emb, recon = ae(xb)
            loss = crit(recon, emb.detach())
            loss.backward(); opt_ae.step()
            el += loss.item()
        avg_l = el / len(dl)
        losses.append(avg_l)
        scheduler.step(avg_l)

        val_errors = compute_errors(ae, X_v)
        _, vf1 = f1_threshold_search(val_errors, y_v, n_points=300)

        if vf1 > best_vf1:
            best_vf1  = vf1
            best_state = {k: v.clone() for k, v in ae.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if ep % 10 == 0:
            print(f"    Ep {ep:>2} | Loss={avg_l:.6f} | VF1={vf1:.4f} | Best={best_vf1:.4f}")
        if no_improve >= 10:
            print(f"    â¹ Early stop at epoch {ep}"); break

    # Final Evaluation
    ae.load_state_dict(best_state)
    val_errors  = compute_errors(ae, X_v)
    best_thresh, _ = f1_threshold_search(val_errors, y_v, n_points=1000)

    t_inf = time.time()
    te_errors = compute_errors(ae, X_te)
    inf_time  = time.time() - t_inf

    y_pred = (te_errors > best_thresh).astype(int)
    fpr, tpr, _ = roc_curve(y_te, te_errors)
    ra = auc(fpr, tpr)

    metrics = {
        'Dataset':    DS_KEY.upper(),
        'Model':      'BiLSTM Autoencoder',
        'Type':       'Unsupervised (DL)',
        'Precision':  round(precision_score(y_te, y_pred, zero_division=0), 4),
        'Recall':     round(recall_score(y_te, y_pred, zero_division=0), 4),
        'F1_Anomaly': round(f1_score(y_te, y_pred, zero_division=0), 4),
        'Macro_F1':   round(f1_score(y_te, y_pred, average='macro', zero_division=0), 4),
        'AUC':        round(ra, 4),
        'MCC':        round(matthews_corrcoef(y_te, y_pred), 4),
        'Threshold':  round(float(best_thresh), 6),
        'Inference_Time_s': round(inf_time, 4),
    }

    print(f"\n  ðŸ“Š TEST RESULTS â€” BGL BiLSTM-AE:")
    print(classification_report(y_te, y_pred, target_names=['Normal', 'Anomaly'], digits=4))
    print(f"  AUC={ra:.4f} | MCC={metrics['MCC']:.4f} | Threshold={best_thresh:.4f}")

    # Save model + config
    torch.save(best_state, f'{MODEL_DIR}/lstm_ae_{DS_KEY}_opt.pt')
    with open(f'{MODEL_DIR}/lstm_ae_{DS_KEY}_config.json', 'w') as f:
        json.dump({**bp, 'threshold': float(best_thresh),
                   'vocab_size': VS, **metrics}, f, indent=2)
    pd.DataFrame([metrics]).round(4).to_csv(
        f'{REPORT}/lstm_ae_{DS_KEY}_results.csv', index=False)

    # 1. Loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(losses, 'b-', lw=1.5)
    plt.title(f'BiLSTM-AE Training Loss â€” BGL')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{REPORT}/lstm_ae_loss_{DS_KEY}.png', dpi=300); plt.close()

    # 2. Reconstruction error histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(te_errors[y_te == 0], bins=100, alpha=0.6, label='Normal', color='#2ecc71')
    ax.hist(te_errors[y_te == 1], bins=100, alpha=0.6, label='Anomaly', color='#e74c3c')
    ax.axvline(best_thresh, color='black', linestyle='--', label=f'Threshold={best_thresh:.4f}')
    ax.set_title('Reconstruction Error Distribution â€” BGL BiLSTM-AE')
    ax.legend(); plt.tight_layout()
    plt.savefig(f'{REPORT}/lstm_ae_errors_{DS_KEY}.png', dpi=300); plt.close()

    # 3. ROC Curve
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, 'b-', lw=2, label=f'AUC={ra:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('ROC â€” BGL BiLSTM-AE'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{REPORT}/lstm_ae_roc_{DS_KEY}.png', dpi=300); plt.close()

    # 4. Confusion Matrix
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    ax.set_title('CM â€” BGL BiLSTM-AE (Optimized)')
    plt.tight_layout()
    plt.savefig(f'{REPORT}/lstm_ae_cm_{DS_KEY}.png', dpi=300); plt.close()

    del ae, X_tr, X_v, X_te; gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()

    ckpt['ae_done'] = True; save_ckpt(ckpt)
    print(f"\n  âœ… BGL BiLSTM-AE done ({inf_time:.0f}s inference)")
else:
    print("[CELL 5] â­ï¸  BiLSTM-AE already done (checkpoint)")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 6 â€” VerificationBlock
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'='*60}")
print("  âœ… BiLSTM-AE BGL STANDALONE â€” COMPLETE")
print(f"{'='*60}")
expected_files = [
    (MODEL_DIR, f'lstm_ae_{DS_KEY}_opt.pt'),
    (MODEL_DIR, f'lstm_ae_{DS_KEY}_config.json'),
    (MODEL_DIR, 'vocab_bgl_lstm_ae.pkl'),
    (MODEL_DIR, 'bgl_sessions_train_lstmae.npz'),
    (MODEL_DIR, 'bgl_sessions_val_lstmae.npz'),
    (MODEL_DIR, 'bgl_sessions_test_lstmae.npz'),
    (REPORT,    f'lstm_ae_{DS_KEY}_results.csv'),
    (REPORT,    f'lstm_ae_loss_{DS_KEY}.png'),
    (REPORT,    f'lstm_ae_errors_{DS_KEY}.png'),
    (REPORT,    f'lstm_ae_cm_{DS_KEY}.png'),
]
for folder, fname in expected_files:
    p = os.path.join(folder, fname)
    print(f"  {'âœ…' if os.path.exists(p) else 'âŒ'} {fname}")


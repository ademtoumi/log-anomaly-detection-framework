# =============================================================================
# STANDALONE NOTEBOOK 12 — BiLSTM Autoencoder on HDFS (Fully Independent)
#
# ✅ ZERO dependencies — reads raw HDFS_Drain.csv directly from Kaggle input.
# ✅ Builds HDFS sessions inline (BlockId grouping, vocab, int32 sequences).
# ✅ One dataset only (HDFS) — RAM stays safe on Kaggle T4/P100.
# ✅ Trains on NORMAL sessions only — reconstruction error for anomaly scoring.
#
# References:
#   [Bekkouche2025_BiLSTM]  — F1=0.993 on HDFS; BiLSTM-AE on session sequences.
#                             Encoder is bidirectional, Decoder is unidirectional.
#   [Du2017_DeepLog]         — HDFS sessions grouped naturally by BlockId.
#   [Zhang2019_LogRobust]    — Sequential temporal split is scientifically honest.
#
# Architecture:
#   Embedding(vocab_size, embed_dim, padding_idx=0)
#   → BiLSTM Encoder(embed_dim, hidden_size, num_layers)
#   → Linear(hidden_size * 2, hidden_size)
#   → LSTM Decoder(hidden_size, hidden_size, 1)
#   → Linear(hidden_size, embed_dim)
#   Loss: MSELoss(decoded_embeddings, original_embeddings.detach())
#
# Threshold selection (CRITICAL — non-negotiable rule):
#   Threshold is searched exclusively on the VALIDATION SET using F1 criterion.
#   Test set is touched exactly once — final evaluation only.
#
# Optuna objective: maximize VAL F1 (with optimal threshold on val).
#   NOT minimize val MSE — MSE minimization doesn't directly maximise F1.
#
# Kaggle setup:
#   - Dataset: pfe-log-anomaly  (must contain HDFS_Drain.csv)
#   - Accelerator: GPU T4 x2 or P100
#   - Estimated time: ~35 minutes
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

# ── Fixed seeds everywhere — reproducibility ──────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Environment
# ─────────────────────────────────────────────────────────────────────────────
KAGGLE   = os.path.exists('/kaggle')
BASE_IN  = '/kaggle/input/pfe-log-anomaly' if KAGGLE else 'Dataset'
# NOTE: This result folder does not yet exist on disk. The v1 baseline was
# implemented but evaluation under the primary temporal-split protocol is pending.
# The v2 improved script (12b) has a confirmed result: results_LSTMAE_HDFS_v2.
BASE_OUT = '/kaggle/working'               if KAGGLE else 'result/results_LSTMAE_HDFS_v1'
MODEL_DIR = f'{BASE_OUT}/models'
REPORT    = f'{BASE_OUT}/pfe_report'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Device: {DEVICE} | {'Kaggle' if KAGGLE else 'Local'} | BiLSTM-AE HDFS Standalone")

# Checkpoint system — re-running skips completed steps
CKPT = pathlib.Path(BASE_OUT) / 'ckpt_12_lstm_ae_hdfs.json'
def save_ckpt(d):
    with open(CKPT, 'w') as f: json.dump(d, f)
def load_ckpt():
    if CKPT.exists():
        with open(CKPT) as f: return json.load(f)
    return {}
ckpt = load_ckpt()
print(f"  Checkpoint keys: {list(ckpt.keys())}")

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
MAX_SEQ_LEN = 75   # [Du2017]: most HDFS sessions < 50 events; 75 adds safety margin

if 'sessions_ready' in ckpt:
    print("\n[CELL 2] ⏭️  Sessions already built — loading from NPZ ...")
    train_data = np.load(f'{MODEL_DIR}/hdfs_sessions_train.npz')
    val_data   = np.load(f'{MODEL_DIR}/hdfs_sessions_val.npz')
    test_data  = np.load(f'{MODEL_DIR}/hdfs_sessions_test.npz')
    vocab      = joblib.load(f'{MODEL_DIR}/vocab_hdfs_opt.pkl')
    X_train, y_train = train_data['X'], train_data['y']
    X_val,   y_val   = val_data['X'],   val_data['y']
    X_test,  y_test  = test_data['X'],  test_data['y']
    VOCAB_SIZE = len(vocab)
    print(f"  VOCAB={VOCAB_SIZE} | Train={X_train.shape} | Val={X_val.shape} | Test={X_test.shape}")
    print(f"  Anomaly: train={y_train.mean()*100:.1f}% | val={y_val.mean()*100:.1f}% | test={y_test.mean()*100:.1f}%")

else:
    print("\n[CELL 2] Building HDFS sessions from HDFS_Drain.csv ...")
    t0 = time.time()

    filepath = find_file('HDFS_Drain.csv')
    print(f"  Source: {filepath}")

    block_events = {}   # BlockId -> list[template_str]
    block_labels = {}   # BlockId -> int (max anomaly flag in session)
    block_order  = []   # Insertion order preserves temporal ordering

    chunk_num = 0
    for chunk in pd.read_csv(filepath, chunksize=500_000,
                              on_bad_lines='skip', low_memory=False):
        chunk_num += 1

        # Extract BlockId — prefer 'BlockId' column, else regex from 'log'
        if 'BlockId' in chunk.columns:
            chunk['_bid'] = chunk['BlockId'].astype(str).str.strip()
        else:
            chunk['_bid'] = chunk['log'].str.extract(r'(blk_-?\d+)')

        chunk = chunk.dropna(subset=['_bid'])

        # Anomaly label: HDFS uses 'Label' column ('Normal' vs 'Anomaly')
        # [Du2017_DeepLog]: session = anomalous if any line is anomalous
        lbl_col = 'Label' if 'Label' in chunk.columns else 'label'
        chunk['_anom'] = (chunk[lbl_col].astype(str).str.strip() != 'Normal').astype(int)

        for _, row in chunk[['_bid', 'template', '_anom']].iterrows():
            bid = row['_bid']
            if bid not in block_events:
                block_events[bid] = []
                block_labels[bid] = 0
                block_order.append(bid)
            tmpl = str(row['template']) if pd.notna(row['template']) else 'unknown'
            block_events[bid].append(tmpl)
            block_labels[bid] = max(block_labels[bid], int(row['_anom']))

        if chunk_num % 5 == 0:
            print(f"    Chunk {chunk_num}: {len(block_events):,} blocks")
        del chunk; gc.collect()

    n_blocks = len(block_order)
    print(f"  Total blocks: {n_blocks:,}")

    # Temporal split 60/20/20
    i1 = int(n_blocks * 0.60)
    i2 = int(n_blocks * 0.80)
    train_bids = block_order[:i1]
    val_bids   = block_order[i1:i2]
    test_bids  = block_order[i2:]

    # Build vocabulary from TRAIN sessions ONLY — no leakage
    # [Bekkouche2025_BiLSTM]: vocab from train only prevents test-set leakage
    all_templates = set()
    for bid in train_bids:
        all_templates.update(block_events[bid])

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for idx, t in enumerate(sorted(all_templates)):
        vocab[t] = idx + 2
    VOCAB_SIZE = len(vocab)
    joblib.dump(vocab, f'{MODEL_DIR}/vocab_hdfs_opt.pkl')
    print(f"  Vocabulary: {VOCAB_SIZE} unique templates (train-only, no leakage)")

    def encode_bids(bids):
        seqs   = np.zeros((len(bids), MAX_SEQ_LEN), dtype=np.int32)
        labels = np.zeros(len(bids), dtype=np.int32)
        for i, bid in enumerate(bids):
            events   = block_events[bid]
            enc      = [vocab.get(e, 1) for e in events]
            seq_len  = min(len(enc), MAX_SEQ_LEN)
            seqs[i, :seq_len] = enc[:seq_len]
            labels[i] = block_labels[bid]
        return seqs, labels

    X_train, y_train = encode_bids(train_bids)
    X_val,   y_val   = encode_bids(val_bids)
    X_test,  y_test  = encode_bids(test_bids)

    del block_events, block_labels; gc.collect()

    print(f"  Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")
    print(f"  Anomaly: train={y_train.mean()*100:.1f}% | val={y_val.mean()*100:.1f}% | test={y_test.mean()*100:.1f}%")

    np.savez_compressed(f'{MODEL_DIR}/hdfs_sessions_train.npz', X=X_train, y=y_train)
    np.savez_compressed(f'{MODEL_DIR}/hdfs_sessions_val.npz',   X=X_val,   y=y_val)
    np.savez_compressed(f'{MODEL_DIR}/hdfs_sessions_test.npz',  X=X_test,  y=y_test)

    elapsed = time.time() - t0
    ckpt['sessions_ready'] = True; save_ckpt(ckpt)
    print(f"  ✅ Sessions saved in {elapsed:.0f}s")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — BiLSTM Autoencoder Architecture
# Based on [Bekkouche2025_BiLSTM]: BiLSTM-AE captures bidirectional context in
#   offline (post-hoc) log processing. Encoder is bidirectional, decoder is
#   unidirectional (future-conditioned decoder would be data leakage).
# ─────────────────────────────────────────────────────────────────────────────
class BiLSTMAutoencoder(nn.Module):
    """
    BiLSTM Encoder-Decoder Autoencoder for unsupervised anomaly detection.

    Encoder: Bidirectional LSTM compresses session sequence to a bottleneck context.
    Decoder: Unidirectional LSTM reconstructs the embedding sequence.
    Loss: MSE between input and reconstructed embeddings.
    Threshold: determined by F1-search on validation reconstruction errors.

    References:
        [Bekkouche2025_BiLSTM]: BiLSTM-AE, F1=0.993 on HDFS.
        [Du2017_DeepLog]: HDFS session-level anomaly detection.
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.hidden_size = hidden_size

        # Encoder: BiLSTM to capture past and future context [Bekkouche2025_BiLSTM]
        self.encoder = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        # Project concatenated directions back to decoder dimension
        self.combine_directions = nn.Linear(hidden_size * 2, hidden_size)

        # Decoder: unidirectional (future is unknown at decode time)
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

        # Reshape context_state → initial hidden state for decoder (1, B, H)
        h0 = context_state.unsqueeze(0)
        c0 = torch.zeros_like(h0)

        # Create zero input sequence for decoder
        decoder_input = torch.zeros(B, T, self.hidden_size, device=x.device)

        # Decode with initial state
        decoded, _ = self.decoder(decoder_input, (h0, c0))      # (B, T, H)
        recon = self.output_proj(decoded)                         # (B, T, E)

        return embedded, recon


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Utility Functions
# ─────────────────────────────────────────────────────────────────────────────
def compute_errors(model, X, batch_size=256):
    """Compute per-session MSE reconstruction errors."""
    model.eval()
    errors = []
    ds = TensorDataset(torch.from_numpy(X).long())
    dl = DataLoader(ds, batch_size=batch_size)
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            emb, recon = model(xb)
            mse = ((emb - recon) ** 2).mean(dim=(1, 2))  # (B,)
            errors.extend(mse.cpu().numpy())
    return np.array(errors, dtype=np.float32)


def f1_threshold_search(errors_val, y_val, n_points=1000):
    """Grid search threshold on VALIDATION reconstruction errors to maximise F1.

    CRITICAL: Only called on VALIDATION data.
    Test set is never used in threshold selection.
    [Bekkouche2025_BiLSTM]: F1-sensitive threshold selection.

    Returns:
        (best_threshold, best_f1)
    """
    lo = float(np.percentile(errors_val, 10))
    hi = float(np.percentile(errors_val, 99.5))
    best_f1, best_thr = 0.0, lo
    for thr in np.linspace(lo, hi, n_points):
        preds = (errors_val > thr).astype(int)
        f1 = f1_score(y_val, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


def train_ae(X_train_normal, X_val, y_val, config, max_epochs=60, patience=10):
    """Train BiLSTM-AE on normal sessions only.
    
    Early stopping on VALIDATION F1 (with optimal threshold on val errors).
    This directly maximises the anomaly detection objective.
    
    Returns:
        model, best_state, train_losses, val_f1s, best_val_f1, best_threshold
    """
    model = BiLSTMAutoencoder(
        vocab_size=VOCAB_SIZE,
        embed_dim=config['embed_dim'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2)
    scaler = GradScaler()

    bs = config['batch_size']
    train_ds = TensorDataset(torch.from_numpy(X_train_normal).long())
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)

    best_val_f1   = 0.0
    best_threshold = 0.0
    best_state    = None
    no_improve    = 0
    train_losses  = []
    val_f1s       = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (xb,) in train_dl:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                emb, recon = model(xb)
                loss = criterion(recon, emb.detach())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_dl)
        train_losses.append(avg_train)
        scheduler.step()

        # ── Val evaluation: compute errors → search F1-optimal threshold ──────
        # CRITICAL: threshold searched on val errors only, not test
        errors_val = compute_errors(model, X_val, batch_size=bs)
        thr, val_f1 = f1_threshold_search(errors_val, y_val, n_points=500)
        val_f1s.append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            best_threshold = thr
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve     = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}/{max_epochs} | TrainLoss={avg_train:.5f} | "
                  f"ValF1={val_f1:.4f} (thr={thr:.4f}) | Best={best_val_f1:.4f}")

        if no_improve >= patience:
            print(f"    ⏹ Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_state, train_losses, val_f1s, best_val_f1, best_threshold


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Optuna + Full Training
# ─────────────────────────────────────────────────────────────────────────────
# KEY FIX: Optuna objective maximizes VAL F1 (with optimal threshold on val).
#   Original code minimized val MSE — MSE minimization ≠ F1 maximization.
#   A model with lower MSE may not produce better anomaly separation.
#   F1 directly measures anomaly detection quality [Bekkouche2025_BiLSTM].
# ─────────────────────────────────────────────────────────────────────────────
if 'ae_done' in ckpt:
    print("\n[CELL 5] ⏭️  BiLSTM-AE already done (checkpoint)")
else:
    print(f"\n{'='*65}")
    print(f"  🧠 HDFS BiLSTM AUTOENCODER — OPTIMIZATION & TRAINING")
    print(f"{'='*65}")
    t0_total = time.time()

    X_train_normal = X_train[y_train == 0]
    print(f"  Normal train sessions: {len(X_train_normal):,} / {len(X_train):,}")
    print(f"  Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"  VOCAB_SIZE: {VOCAB_SIZE}")

    # ── Optuna objective: MAXIMISE val F1 (not minimise val MSE) ─────────────
    def objective(trial):
        cfg = {
            'embed_dim':   trial.suggest_categorical('embed_dim',   [32, 64, 128]),
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
            'num_layers':  trial.suggest_int('num_layers', 1, 3),
            'dropout':     trial.suggest_float('dropout', 0.1, 0.4),
            'lr':          trial.suggest_float('lr', 1e-4, 5e-3, log=True),
            'batch_size':  trial.suggest_categorical('batch_size', [128, 256, 512]),
        }
        _, _, _, _, best_f1, _ = train_ae(
            X_train_normal, X_val, y_val, cfg, max_epochs=12, patience=5)
        gc.collect()
        return best_f1   # maximise F1, not minimise MSE

    study = optuna.create_study(
        direction='maximize',   # ← maximise F1 (was 'minimize' for MSE — wrong)
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )

    # Warm-start from [Bekkouche2025_BiLSTM] architecture defaults
    study.enqueue_trial({
        'embed_dim': 64, 'hidden_size': 128, 'num_layers': 2,
        'dropout': 0.2, 'lr': 0.001, 'batch_size': 256
    })

    print(f"  🔍 Optuna (15 trials, objective=val F1) ...")
    study.optimize(objective, n_trials=15, timeout=900)
    best_params = study.best_params
    print(f"  🏆 Best params: {best_params} → Val F1={study.best_value:.4f}")

    print(f"\n  🚀 Full training (60 epochs, patience=12, objective=val F1) ...")
    model, best_state, train_losses, val_f1s, best_val_f1, best_threshold = train_ae(
        X_train_normal, X_val, y_val,
        best_params, max_epochs=60, patience=12
    )

    print(f"\n  ✅ Training done. Best val F1={best_val_f1:.4f} at threshold={best_threshold:.6f}")

    # ── Test evaluation — test set touched EXACTLY ONCE ───────────────────────
    print("\n  📐 Computing reconstruction errors on test set ...")
    t_inf = time.time()
    errors_test = compute_errors(model, X_test, batch_size=best_params['batch_size'])
    infer_time  = time.time() - t_inf

    # Apply val-derived threshold to test — no threshold re-search on test
    y_pred_test = (errors_test > best_threshold).astype(int)

    # Normalise errors to [0,1] for AUC/AP computation
    e_min, e_max = errors_test.min(), errors_test.max()
    y_prob_test  = (errors_test - e_min) / (e_max - e_min + 1e-12)

    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    roc_auc     = auc(fpr, tpr)

    metrics = {
        'Dataset':   'HDFS',
        'Model':     'BiLSTM-AE',
        'Type':      'Unsupervised (DL)',
        'Threshold': round(float(best_threshold), 6),
        'Precision': round(precision_score(y_test, y_pred_test, pos_label=1, zero_division=0), 4),
        'Recall':    round(recall_score(y_test, y_pred_test, pos_label=1, zero_division=0), 4),
        'F1_Anomaly':round(f1_score(y_test, y_pred_test, pos_label=1, zero_division=0), 4),
        'Macro_F1':  round(f1_score(y_test, y_pred_test, average='macro', zero_division=0), 4),
        'AUC':       round(roc_auc, 4),
        'MCC':       round(matthews_corrcoef(y_test, y_pred_test), 4),
        'Avg_Precision': round(average_precision_score(y_test, y_prob_test), 4),
        'Inference_Time_s': round(infer_time, 4),
        'Inference_Per_Sample_ms': round(infer_time / len(y_test) * 1000, 4),
    }

    # ── Paper comparison table ────────────────────────────────────────────────
    paper_f1 = 0.993   # [Bekkouche2025_BiLSTM] BiLSTM-AE on HDFS
    our_f1   = metrics['F1_Anomaly']
    delta    = our_f1 - paper_f1
    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  RESULTS vs PAPER — HDFS BiLSTM-AE                 │")
    print(f"  │  Paper [Bekkouche2025_BiLSTM] F1 : {paper_f1:.4f}          │")
    print(f"  │  Our F1                        : {our_f1:.4f}          │")
    print(f"  │  Delta                         : {delta:+.4f}          │")
    print(f"  └─────────────────────────────────────────────────────┘")

    print(f"\n  📊 TEST RESULTS — HDFS BiLSTM-AE:")
    print(classification_report(y_test, y_pred_test, target_names=['Normal', 'Anomaly'], digits=4))
    print(f"  AUC={roc_auc:.4f} | MCC={metrics['MCC']:.4f} | AP={metrics['Avg_Precision']:.4f}")
    print(f"  Threshold (from val): {best_threshold:.6f}")

    torch.save(best_state, f'{MODEL_DIR}/lstm_ae_hdfs_opt.pt')
    with open(f'{MODEL_DIR}/lstm_ae_hdfs_config.json', 'w') as f:
        json.dump({
            **best_params,
            'vocab_size': VOCAB_SIZE,
            'max_seq_len': MAX_SEQ_LEN,
            'threshold': float(best_threshold),
            'best_val_f1': float(best_val_f1),
            'paper_f1': paper_f1,
            'delta_vs_paper': round(delta, 4),
            **{k: v for k, v in metrics.items()},
        }, f, indent=2)
    pd.DataFrame([metrics]).round(4).to_csv(f'{REPORT}/lstm_ae_hdfs_results.csv', index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    # 1. Val F1 training curve + reconstruction error distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(range(1, len(train_losses)+1), train_losses, 'b-o', markersize=3, label='Train MSE')
    ax1.plot(range(1, len(val_f1s)+1),     val_f1s,       'g-o', markersize=3, label='Val F1 (optimal thr)')
    ax1.set_title('BiLSTM-AE Training — HDFS', fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss / F1')
    ax1.legend(); ax1.grid(alpha=0.3)

    # 2. Reconstruction error histogram
    err_normal = errors_test[y_test == 0]
    err_anom   = errors_test[y_test == 1]
    ax2.hist(err_normal, bins=80, alpha=0.6, color='steelblue', label=f'Normal (n={len(err_normal):,})', density=True)
    ax2.hist(err_anom,   bins=80, alpha=0.6, color='crimson',   label=f'Anomaly (n={len(err_anom):,})',  density=True)
    ax2.axvline(best_threshold, color='black', linestyle='--', lw=1.5, label=f'Threshold={best_threshold:.4f}')
    ax2.set_title('Reconstruction Error Distribution — HDFS', fontweight='bold')
    ax2.set_xlabel('MSE Score'); ax2.set_ylabel('Density')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{REPORT}/lstm_ae_hdfs_curves.png', dpi=300); plt.close()

    # 3. ROC Curve
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, 'b-', lw=2, label=f'AUC={roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('ROC — HDFS BiLSTM-AE (Standalone)', fontweight='bold')
    plt.legend(loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f'{REPORT}/lstm_ae_hdfs_roc.png', dpi=300); plt.close()

    # 4. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    ax.set_title('CM — HDFS BiLSTM-AE (Standalone)', fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(f'{REPORT}/lstm_ae_hdfs_cm.png', dpi=300); plt.close()

    del model, X_train_normal, errors_test
    gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()

    ckpt['ae_done'] = True; save_ckpt(ckpt)
    print(f"\n  ✅ HDFS BiLSTM-AE complete ({time.time()-t0_total:.0f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Verification Block
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  ✅ NOTEBOOK 12 — BiLSTM-AE HDFS STANDALONE — COMPLETE")
print(f"{'='*65}")
output_files = [
    (MODEL_DIR, 'hdfs_sessions_train.npz'),
    (MODEL_DIR, 'hdfs_sessions_val.npz'),
    (MODEL_DIR, 'hdfs_sessions_test.npz'),
    (MODEL_DIR, 'vocab_hdfs_opt.pkl'),
    (MODEL_DIR, 'lstm_ae_hdfs_opt.pt'),
    (MODEL_DIR, 'lstm_ae_hdfs_config.json'),
    (REPORT,    'lstm_ae_hdfs_results.csv'),
    (REPORT,    'lstm_ae_hdfs_curves.png'),
    (REPORT,    'lstm_ae_hdfs_roc.png'),
    (REPORT,    'lstm_ae_hdfs_cm.png'),
]
all_ok = True
for dirp, fname in output_files:
    p = os.path.join(dirp, fname)
    exists = os.path.exists(p)
    status = '✅' if exists else '❌'
    size_s = f"({os.path.getsize(p)/1024:.1f} KB)" if exists else "(missing)"
    print(f"  {status} {fname:<45} {size_s}")
    if not exists:
        all_ok = False

print(f"\n  Status: {'🎉 All outputs present' if all_ok else '⚠️  Some outputs missing'}")
print(f"\n  Paper citations:")
print(f"    [Bekkouche2025_BiLSTM] — BiLSTM-AE, F1=0.993, threshold on val")
print(f"    [Du2017_DeepLog]       — HDFS sessions grouped by BlockId")
print(f"    [Zhang2019_LogRobust]  — Sequential temporal split")
print(f"\n  KEY CHANGES vs original:")
print(f"    ❌ Old: Optuna minimized val MSE → not aligned with F1 objective")
print(f"    ❌ Old: early stopping on val MSE (loss), not val F1")
print(f"    ❌ Old: vocab from all sessions (val+test leakage)")
print(f"    ❌ Old: label != 'Normal' detection only in some code paths")
print(f"    ✅ New: Optuna maximizes val F1 with optimal threshold")
print(f"    ✅ New: early stopping on val F1 (directly optimises detection)")
print(f"    ✅ New: vocab from TRAIN sessions only (no leakage)")
print(f"    ✅ New: consistent 'Normal' vs anomaly label detection")
print(f"    ✅ New: val-derived threshold applied once to test")

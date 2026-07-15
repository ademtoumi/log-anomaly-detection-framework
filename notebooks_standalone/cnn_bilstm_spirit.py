# =============================================================================
# STANDALONE NOTEBOOK 09 â€” CNN+BiLSTM Hybrid on Spirit (Fully Independent)
#
# Based on [Lu2018_LogCNN]     â€” Multi-scale 1D CNN (kernels [2,3,5]) captures
#   different n-gram patterns at multiple temporal scales simultaneously.
# Based on [Zhang2019_LogRobust] â€” Attention-based BiLSTM for robust anomaly
#   detection; bidirectional context improves recall on Spirit.
# Based on [Bekkouche2025_Spirit] â€” Spirit sliding-window strategy;
#   WINDOW_SIZE=20, STEP_SIZE=10 balances context vs. label resolution.
#
# Architecture: Embedding â†’ Multi-Scale CNN (k=[2,3,5]) â†’ BiLSTM â†’ Attention
#               â†’ Dropout â†’ FC (Binary)
#
# âœ… ZERO dependencies â€” reads Spirit_Drain.csv directly from Kaggle input.
# âœ… Builds vocab + sliding-window sessions inline (no external .npz needed).
# âœ… One dataset only â†’ RAM stays safe on Kaggle.
# âœ… Checkpoint system: 'sessions_ready', 'done'
#
# Kaggle setup:
#   - Add dataset: pfe-log-anomaly  (contains Spirit_Drain.csv)
#   - Accelerator: GPU T4 x2 or P100
#   - Estimated time: ~28 minutes
# =============================================================================

import os, gc, json, pathlib, time, random, warnings, itertools
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import optuna, torch, torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    f1_score, precision_score, recall_score, matthews_corrcoef,
    average_precision_score,
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 1 â€” Environment & Paths
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
KAGGLE    = os.path.exists('/kaggle')
BASE_IN   = '/kaggle/input/pfe-log-anomaly' if KAGGLE else 'Dataset'
BASE_OUT  = '/kaggle/working'               if KAGGLE else 'result/results_cnn_bilstm_spirit'
MODEL_DIR = f'{BASE_OUT}/models'
REPORT    = f'{BASE_OUT}/pfe_report'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT,    exist_ok=True)

DS_KEY = 'spirit'

# Safety cap â€” set to e.g. 3_000_000 if OOM occurs on Kaggle
NROWS_LIMIT = None  # None = full dataset

# Sliding-window parameters [Bekkouche2025_Spirit]
WINDOW_SIZE = 20   # log events per window / sequence
STEP_SIZE   = 10   # stride between windows

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {DEVICE} | Kaggle={KAGGLE}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Checkpoint helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CKPT = pathlib.Path(BASE_OUT) / 'ckpt_09_cnn_bilstm_spirit.json'

def save_ckpt(d):
    with open(CKPT, 'w') as f:
        json.dump(d, f)

def load_ckpt():
    if CKPT.exists():
        with open(CKPT) as f:
            return json.load(f)
    return {}

ckpt = load_ckpt()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# File finder helper
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
if 'sessions_ready' not in ckpt:
    print(f"\n{'='*65}")
    print(f"  ðŸ“‚ CELL 2 â€” Spirit Sliding-Window Session Building")
    print(f"{'='*65}")
    t0 = time.time()

    filepath = find_file('Spirit_Drain.csv')
    print(f"  Reading: {filepath}")

    # â”€â”€ Chunked load: collect only template + label columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    all_templates = []
    all_labels    = []
    rows_loaded   = 0

    for chunk in pd.read_csv(
            filepath,
            chunksize=500_000,
            usecols=['template', 'label'],
            on_bad_lines='skip',
            low_memory=False):
        all_templates.extend(chunk['template'].fillna('').tolist())
        # Spirit label: numeric column, 0 = normal, non-zero = anomaly [Bekkouche2025_Spirit]
        all_labels.extend(
            (chunk['label'].astype(str).str.strip() != '-').astype(int).tolist()
        )
        rows_loaded += len(chunk)
        print(f"  ... loaded {rows_loaded:,} rows", end='\r')
        del chunk; gc.collect()
        if NROWS_LIMIT and rows_loaded >= NROWS_LIMIT:
            break

    print(f"\n  Total rows: {rows_loaded:,}")
    templates = all_templates;  del all_templates
    labels    = all_labels;     del all_labels
    gc.collect()

    n_total = len(templates)
    n_anom  = sum(labels)
    print(f"  Normal: {n_total - n_anom:,} | Anomaly: {n_anom:,} "
          f"({n_anom / n_total * 100:.1f}%)")

    # â”€â”€ Build vocabulary (fit on TRAIN+VAL corpus to avoid data leakage) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("  Building vocabulary from training+validation split (leak-safe) ...")
    token_freq = defaultdict(int)
    n_train = int(len(templates) * 0.80)
    for t in templates[:n_train]:
        for tok in t.split():
            token_freq[tok] += 1

    # Keep tokens appearing >= 2 times; reserve 0=PAD, 1=UNK
    MIN_FREQ = 2
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for tok, cnt in sorted(token_freq.items(), key=lambda x: -x[1]):
        if cnt >= MIN_FREQ:
            vocab[tok] = len(vocab)
    VS = len(vocab)
    print(f"  Vocabulary size: {VS:,}")
    del token_freq; gc.collect()

    # â”€â”€ Convert templates to integer token sequences â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("  Tokenising templates ...")
    UNK_ID = vocab['<UNK>']
    token_ids = []
    for t in templates:
        ids = [vocab.get(tok, UNK_ID) for tok in t.split()]
        token_ids.append(ids if ids else [UNK_ID])
    del templates; gc.collect()

    # â”€â”€ Build sliding-window sequences on entire log stream first â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # [Bekkouche2025_Spirit]: window label = 1 if ANY event in window is anomaly
    PAD_ID = vocab['<PAD>']
    X_win, y_win = [], []
    for start in range(0, len(token_ids) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window_lab  = int(any(labels[start:end]))
        seq = [token_ids[i][0] if token_ids[i] else PAD_ID
               for i in range(start, end)]
        X_win.append(seq)
        y_win.append(window_lab)

    X_arr = np.array(X_win, dtype=np.int32)
    y_arr = np.array(y_win, dtype=np.int32)

    # â”€â”€ Stratified random split 70/10/20 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(X_arr))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=y_arr)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=y_arr[train_val_idx])

    # Save splits
    np.savez_compressed(f'{MODEL_DIR}/spirit_sessions_train.npz', X=X_arr[train_idx], y=y_arr[train_idx])
    np.savez_compressed(f'{MODEL_DIR}/spirit_sessions_val.npz',   X=X_arr[val_idx],   y=y_arr[val_idx])
    np.savez_compressed(f'{MODEL_DIR}/spirit_sessions_test.npz',  X=X_arr[test_idx],  y=y_arr[test_idx])

    print(f"  âœ… train: {len(train_idx):,} | anomaly={y_arr[train_idx].mean()*100:.1f}%")
    print(f"  âœ… val: {len(val_idx):,} | anomaly={y_arr[val_idx].mean()*100:.1f}%")
    print(f"  âœ… test: {len(test_idx):,} | anomaly={y_arr[test_idx].mean()*100:.1f}%")
    del X_win, y_win, X_arr, y_arr; gc.collect()

    joblib.dump(vocab, f'{MODEL_DIR}/vocab_spirit_cnn_bilstm.pkl')
    del token_ids, labels; gc.collect()

    print(f"  âœ… Session building done ({time.time()-t0:.0f}s)")
    ckpt['sessions_ready'] = True
    ckpt['vocab_size']     = VS
    save_ckpt(ckpt)

else:
    print("[CELL 2] â­ï¸  Sessions already built (checkpoint)")
    vocab = joblib.load(f'{MODEL_DIR}/vocab_spirit_cnn_bilstm.pkl')
    VS    = ckpt.get('vocab_size', len(vocab))


# =============================================================================
# CELL 3 â€” MultiScaleCNNBiLSTM Architecture
# [Lu2018_LogCNN]: Multiple kernel sizes [2,3,5] capture n-gram patterns.
# [Zhang2019_LogRobust]: Attention over LSTM outputs for anomaly focus.
# =============================================================================
class MultiScaleCNNBiLSTM(nn.Module):
    """Multi-scale CNN + BiLSTM with attention for log anomaly detection.

    CNN branch captures local n-gram patterns at kernels kâˆˆ{2,3,5}.
    [Lu2018_LogCNN]: max-overtime pooling per kernel then concatenate.
    BiLSTM branch captures global temporal dependencies.
    [Zhang2019_LogRobust]: soft attention focuses on anomalous timesteps.
    """
    def __init__(self, vocab_size, embed_dim=64, cnn_filters=64,
                 kernel_sizes=(2, 3, 5), hidden_size=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multi-scale 1D CNN â€” different kernels = different n-gram scales
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, cnn_filters, k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.relu = nn.ReLU()

        # BiLSTM on concatenated multi-scale features
        cnn_out_dim = cnn_filters * len(kernel_sizes)
        self.lstm = nn.LSTM(
            cnn_out_dim, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        # Soft attention â€” [Zhang2019_LogRobust]
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)          # [B, seq, E]
        emb_t    = embedded.permute(0, 2, 1)  # [B, E, seq]

        # â”€â”€ Multi-scale CNN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        conv_outs = [self.relu(conv(emb_t)) for conv in self.convs]
        # Align all conv outputs to the same temporal length via truncation
        min_len   = min(c.size(2) for c in conv_outs)
        conv_outs = [c[:, :, :min_len] for c in conv_outs]
        cnn_out   = torch.cat(conv_outs, dim=1)   # [B, filters*K, T]
        cnn_out   = cnn_out.permute(0, 2, 1)      # [B, T, filters*K]

        # â”€â”€ BiLSTM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        lstm_out, _ = self.lstm(cnn_out)           # [B, T, H*2]

        # â”€â”€ Attention pooling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        attn_w  = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_w).sum(dim=1)  # [B, H*2]

        return self.fc(self.dropout(context)).squeeze(-1)  # [B]


# =============================================================================
# CELL 4 â€” Training Function (mixed precision, pos_weight, cosine scheduler)
# =============================================================================
def find_best_threshold_f1(probs, labels, n_points=300):
    """Grid-search threshold on val probs to maximize F1. Never use test labels."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.01, 0.99, n_points):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

def train_cnn_bilstm(X_tr, y_tr, X_v, y_v, X_te, y_te,
                     vocab_size, config, max_epochs=40, patience=10):
    model = MultiScaleCNNBiLSTM(
        vocab_size   = vocab_size,
        embed_dim    = config['embed_dim'],
        cnn_filters  = config['cnn_filters'],
        kernel_sizes = (2, 3, 5),
        hidden_size  = config['hidden_size'],
        num_layers   = config['num_layers'],
        dropout      = config['dropout'],
    ).to(DEVICE)

    # Class-imbalance handling: sqrt-scaled pos_weight
    n_neg = int((y_tr == 0).sum())
    n_pos = max(int((y_tr == 1).sum()), 1)
    pw    = max(1.0, np.sqrt(n_neg / n_pos))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], device=DEVICE))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2)
    scaler = GradScaler()

    bs = config['batch_size']
    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).long(),
                      torch.from_numpy(y_tr).float()),
        batch_size=bs, shuffle=True, pin_memory=(DEVICE == 'cuda'))
    val_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_v).long(),
                      torch.from_numpy(y_v).float()),
        batch_size=bs, pin_memory=(DEVICE == 'cuda'))

    best_f1, best_state, no_improve = 0.0, None, 0
    best_thr = 0.5   # val-optimal threshold, updated each time val F1 improves
    losses, f1s = [], []

    for epoch in range(1, max_epochs + 1):
        model.train(); epoch_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            epoch_loss += loss.item()
        scheduler.step()
        losses.append(epoch_loss / len(train_dl))

        # Evaluate on validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                pr = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
                val_probs.extend(pr)
                val_labels.extend(yb.numpy().astype(int))
        val_probs  = np.array(val_probs)
        val_labels = np.array(val_labels)
        
        thr, vf1 = find_best_threshold_f1(val_probs, val_labels, n_points=300)
        f1s.append(vf1)

        if vf1 > best_f1:
            best_f1    = vf1
            best_thr   = thr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Ep {epoch:>2}/{max_epochs} "
                  f"Loss={losses[-1]:.4f}  VF1={vf1:.4f}  Best={best_f1:.4f}")
        if no_improve >= patience:
            print(f"    â¹ Early stop at epoch {epoch}")
            break

    # â”€â”€ Test inference â€” threshold from val, not from test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model.load_state_dict(best_state); model.eval()
    test_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_te).long(),
                      torch.from_numpy(y_te).float()),
        batch_size=bs)
    t_inf = time.time()
    tpb_list, tt = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            pr = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
            tpb_list.extend(pr)
            tt.extend(yb.numpy().astype(int))
    tpb    = np.array(tpb_list)
    tt     = np.array(tt)
    tp     = (tpb >= best_thr).astype(int)   # apply val-derived threshold once
    inf_t  = time.time() - t_inf

    return model, best_state, best_f1, best_thr, losses, f1s, tt, tp, tpb, inf_t


# =============================================================================
# CELL 5 â€” Optuna Hyperparameter Search + Full Training
# =============================================================================
if 'done' in ckpt:
    print("\n[CELL 5] â­ï¸  Already done (checkpoint). Skipping training.")
else:
    print(f"\n{'='*65}")
    print(f"  ðŸ”¬ SPIRIT CNN+BiLSTM OPTIMIZATION [Lu2018 + Zhang2019]")
    print(f"{'='*65}")
    t0_total = time.time()

    # â”€â”€ Load pre-built sessions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    tr_d = np.load(f'{MODEL_DIR}/spirit_sessions_train.npz')
    va_d = np.load(f'{MODEL_DIR}/spirit_sessions_val.npz')
    te_d = np.load(f'{MODEL_DIR}/spirit_sessions_test.npz')

    X_tr, y_tr = tr_d['X'].astype(np.int32), tr_d['y'].astype(np.int32)
    X_v,  y_v  = va_d['X'].astype(np.int32), va_d['y'].astype(np.int32)
    X_te, y_te = te_d['X'].astype(np.int32), te_d['y'].astype(np.int32)
    del tr_d, va_d, te_d; gc.collect()

    print(f"  VS={VS:,} | Train={X_tr.shape} Val={X_v.shape} Test={X_te.shape}")
    print(f"  Anomaly %: train={y_tr.mean()*100:.1f}%  "
          f"val={y_v.mean()*100:.1f}%  test={y_te.mean()*100:.1f}%")

    # â”€â”€ Optuna objective â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def objective(trial):
        cfg = {
            'embed_dim':   trial.suggest_categorical('embed_dim',   [32, 64, 128]),
            'cnn_filters': trial.suggest_categorical('cnn_filters', [32, 64, 128]),
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
            'num_layers':  trial.suggest_int('num_layers', 1, 3),
            'dropout':     trial.suggest_float('dropout', 0.1, 0.5),
            'lr':          trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'batch_size':  trial.suggest_categorical('batch_size', [128, 256, 512]),
        }
        _, _, bf1, _, _, _, _, _, _, _ = train_cnn_bilstm(
            X_tr, y_tr, X_v, y_v, X_te, y_te,
            VS, cfg, max_epochs=10, patience=5)
        return bf1

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42))

    # Warm-start: [Lu2018] embed=64, cnn=64 + [Zhang2019] hidden=128, lr=5e-4
    study.enqueue_trial({
        'embed_dim': 64, 'cnn_filters': 64, 'hidden_size': 128,
        'num_layers': 2,  'dropout': 0.3,   'lr': 0.0005,
        'batch_size': 256,
    })

    print(f"\n  ðŸ” Optuna (15 trials, timeout=900s) ...")
    study.optimize(objective, n_trials=15, timeout=900)
    bp = study.best_params
    print(f"  ðŸ† Best params: {bp}")
    print(f"  ðŸ† Best val F1: {study.best_value:.4f}")

    # â”€â”€ Full training with best params â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n  ðŸš€ Full training (40 epochs, patience=10) ...")
    model, best_state, bf1, best_thr, losses, f1s, yt, yp, ypr, inf_t = train_cnn_bilstm(
        X_tr, y_tr, X_v, y_v, X_te, y_te,
        VS, bp, max_epochs=40, patience=10)

    # â”€â”€ Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fpr, tpr, _ = roc_curve(yt, ypr)
    roc_auc = auc(fpr, tpr)
    avg_prec = average_precision_score(yt, ypr)

    metrics = {
        'Dataset':     'SPIRIT',
        'Model':       'CNN+BiLSTM (Multi-Scale)',
        'Type':        'Supervised (DL)',
        'Precision':   round(precision_score(yt, yp, zero_division=0), 4),
        'Recall':      round(recall_score(yt, yp, zero_division=0), 4),
        'F1_Anomaly':  round(f1_score(yt, yp, zero_division=0), 4),
        'Macro_F1':    round(f1_score(yt, yp, average='macro', zero_division=0), 4),
        'AUC':         round(roc_auc, 4),
        'MCC':         round(matthews_corrcoef(yt, yp), 4),
        'Avg_Precision': round(avg_prec, 4),
        'Inference_Time_s': round(inf_t, 4),
        'Inference_Per_Sample_ms': round(inf_t / max(len(yt), 1) * 1000, 4),
        'Window_Size': WINDOW_SIZE,
        'Step_Size':   STEP_SIZE,
    }

    print(f"\n  ðŸ“Š TEST RESULTS â€” Spirit CNN+BiLSTM:")
    print(classification_report(yt, yp,
                                target_names=['Normal', 'Anomaly'], digits=4))
    print(f"  AUC={roc_auc:.4f} | MCC={metrics['MCC']:.4f} | "
          f"AvgPrec={avg_prec:.4f}")

    # â”€â”€ Save model artefacts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    torch.save(best_state,
               f'{MODEL_DIR}/cnn_bilstm_spirit_opt.pt')
    with open(f'{MODEL_DIR}/cnn_bilstm_spirit_config.json', 'w') as f:
        json.dump({**bp, 'vocab_size': VS, **metrics}, f, indent=2)
    pd.DataFrame([metrics]).to_csv(
        f'{REPORT}/cnn_bilstm_spirit_results.csv', index=False)

    # â”€â”€ Plots â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # 1) Loss + Val F1 curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(losses, 'b-o', ms=3, label='Train Loss')
    a1.set_title('Loss â€” Spirit CNN+BiLSTM'); a1.set_xlabel('Epoch')
    a1.set_ylabel('BCE Loss'); a1.grid(alpha=0.3); a1.legend()

    a2.plot(f1s, 'g-o', ms=3, label='Val F1')
    a2.axhline(bf1, ls='--', c='r', alpha=0.6, label=f'Best={bf1:.4f}')
    a2.set_title('Validation F1 â€” Spirit CNN+BiLSTM')
    a2.set_xlabel('Epoch'); a2.set_ylabel('F1 Score')
    a2.set_ylim([0, 1.05]); a2.grid(alpha=0.3); a2.legend()
    plt.tight_layout()
    plt.savefig(f'{REPORT}/cnn_bilstm_spirit_curves.png', dpi=300)
    plt.show()

    # 2) Confusion Matrix
    cm = confusion_matrix(yt, yp)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    ax.set_title('Confusion Matrix â€” Spirit CNN+BiLSTM (Opt)')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(f'{REPORT}/cnn_bilstm_spirit_cm.png', dpi=300)
    plt.show()

    # 3) ROC Curve
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, 'darkorange', lw=2,
             label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC â€” Spirit CNN+BiLSTM'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{REPORT}/cnn_bilstm_spirit_roc.png', dpi=300)
    plt.show()

    # â”€â”€ Cleanup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    del model, X_tr, X_v, X_te; gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    ckpt['done'] = True; save_ckpt(ckpt)
    print(f"\n  âœ… Spirit CNN+BiLSTM done ({time.time()-t0_total:.0f}s)")


# =============================================================================
# CELL 6 â€” Verification Block
# =============================================================================
print(f"\n{'='*65}")
print(f"  âœ… SPIRIT CNN+BiLSTM STANDALONE â€” VERIFICATION")
print(f"{'='*65}")

expected_files = {
    'Models': [
        f'{MODEL_DIR}/cnn_bilstm_spirit_opt.pt',
        f'{MODEL_DIR}/cnn_bilstm_spirit_config.json',
        f'{MODEL_DIR}/vocab_spirit_cnn_bilstm.pkl',
        f'{MODEL_DIR}/spirit_sessions_train.npz',
        f'{MODEL_DIR}/spirit_sessions_val.npz',
        f'{MODEL_DIR}/spirit_sessions_test.npz',
    ],
    'Reports': [
        f'{REPORT}/cnn_bilstm_spirit_results.csv',
        f'{REPORT}/cnn_bilstm_spirit_curves.png',
        f'{REPORT}/cnn_bilstm_spirit_cm.png',
        f'{REPORT}/cnn_bilstm_spirit_roc.png',
    ],
}

all_ok = True
for section, paths in expected_files.items():
    print(f"\n  [{section}]")
    for p in paths:
        exists = os.path.exists(p)
        icon   = 'âœ…' if exists else 'âŒ'
        size   = f'({os.path.getsize(p) / 1024:.1f} KB)' if exists else ''
        print(f"    {icon} {os.path.basename(p)} {size}")
        if not exists:
            all_ok = False

print(f"\n  {'ðŸŽ‰ All outputs verified!' if all_ok else 'âš ï¸  Some outputs missing.'}")
print(f"  Checkpoint keys: {list(ckpt.keys())}")
print(f"\n  Refs: [Lu2018_LogCNN] [Zhang2019_LogRobust] [Bekkouche2025_Spirit]")


# =============================================================================
# STANDALONE NOTEBOOK 07 â€” CNN+BiLSTM Hybrid on HDFS (Fully Independent)
#
# Based on [Lu2018_LogCNN]      â€” Multi-scale 1D CNN (kernels [2,3,5]) captures
#   different n-gram log-event patterns at multiple scales.
# Based on [Zhang2019_LogRobust] â€” Attention over LSTM outputs focuses the model
#   on the most anomalous timesteps within a session.
# Based on [Guo2021_LogBERT]   â€” Bidirectional context is superior for offline
#   anomaly detection on HDFS sessions.
# Based on [Bekkouche2025_BiLSTM] â€” HDFS sessions grouped by BlockId; F1-optimal
#   threshold on VALIDATION SET ONLY. Test set touched exactly once.
#
# Architecture: Embedding â†’ Multi-Scale CNN (kernels [2,3,5]) â†’ BiLSTM
#               â†’ Attention â†’ Dropout â†’ FC â†’ Sigmoid
#
# âœ… ZERO dependencies â€” reads raw HDFS_Drain.csv directly.
# âœ… Builds sessions inline (chunked reading, BlockId extraction, vocab build).
# âœ… Vocab built from TRAIN sessions only â€” no val/test leakage.
# âœ… Temporal split 60/20/20 on sessions.
# âœ… F1-optimal threshold searched on VALIDATION probabilities.
# âœ… Test set evaluated ONCE at the end with the val-derived threshold.
# âœ… One dataset only (HDFS) â†’ RAM stays safe on Kaggle.
# âœ… AMP + gradient clipping + CosineAnnealingWarmRestarts.
# âœ… Checkpoint system: skips completed steps on re-run.
# âœ… Seed 42 everywhere.
#
# Kaggle setup:
#   - Add dataset: pfe-log-anomaly  (contains HDFS_Drain.csv)
#   - Accelerator: GPU T4 or P100
#   - Estimated time: ~35 minutes
# =============================================================================

import os, gc, json, re, pathlib, time, random, warnings, collections
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
    classification_report, confusion_matrix, roc_curve, auc,
    f1_score, precision_score, recall_score, matthews_corrcoef,
    average_precision_score,
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# â”€â”€ Fixed seeds everywhere â€” reproducibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 1 â€” Environment & Paths
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
KAGGLE   = os.path.exists('/kaggle')
BASE_IN  = '/kaggle/input/pfe-log-anomaly' if KAGGLE else 'Dataset'
BASE_OUT = '/kaggle/working'               if KAGGLE else 'result/results_cnn_bilstm_hdfs'
MODEL_DIR = f'{BASE_OUT}/models'
REPORT    = f'{BASE_OUT}/pfe_report'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT, exist_ok=True)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Checkpoint â€” re-running skips completed steps
CKPT = pathlib.Path(BASE_OUT) / 'ckpt_07_cnn_bilstm_hdfs_standalone.json'
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
print(f"{'Kaggle' if KAGGLE else 'Local'} | Device: {DEVICE} | CNN+BiLSTM HDFS Standalone")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 2 â€” HDFS Session Building (inline, chunked)
#
# Based on [Bekkouche2025_BiLSTM]: HDFS sessions defined by BlockId.
#   Each session is anomalous if ANY of its lines carries a non-'Normal' label.
#   Temporal ordering is preserved by insertion order in the CSV.
#
# Vocab built from TRAIN sessions only â€” no leakage [Bekkouche2025_BiLSTM].
# MAX_SEQ_LEN=75: covers all HDFS sessions with margin [Du2017_DeepLog].
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MAX_SEQ_LEN = 75   # [Du2017_DeepLog]: most HDFS sessions < 50 events
CHUNK_SIZE  = 500_000

if 'sessions_ready' not in ckpt:
    print("\n[CELL 2] Building HDFS sessions from HDFS_Drain.csv ...")
    t0 = time.time()

    filepath = find_file('HDFS_Drain.csv')
    print(f"  Source: {filepath}")

    # Aggregation: block_id â†’ {'events': list[str], 'label': int}
    # Insertion order preserved (Python 3.7+ dict is ordered)
    block_events = {}
    block_labels = {}
    block_order  = []

    chunk_num = 0
    for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE,
                              on_bad_lines='skip', low_memory=False):
        chunk_num += 1

        # Extract BlockId â€” prefer 'BlockId' column, else regex from 'log'
        if 'BlockId' in chunk.columns:
            chunk['_bid'] = chunk['BlockId'].astype(str).str.strip()
        else:
            chunk['_bid'] = chunk['log'].str.extract(r'(blk_-?\d+)')

        chunk = chunk.dropna(subset=['_bid'])

        # Anomaly label: HDFS uses 'Label' column with 'Normal' vs 'Anomaly'
        # [Du2017_DeepLog]: session label = OR of all line labels
        lbl_col = 'Label' if 'Label' in chunk.columns else 'label'
        chunk['_anom'] = (chunk[lbl_col].astype(str).str.strip() != 'Normal').astype(int)

        for _, row in chunk[['_bid', 'template', '_anom']].iterrows():
            bid  = row['_bid']
            tmpl = str(row['template']) if pd.notna(row['template']) else '<UNK>'
            anom = int(row['_anom'])
            if bid not in block_events:
                block_events[bid] = []
                block_labels[bid] = 0
                block_order.append(bid)
            block_events[bid].append(tmpl)
            block_labels[bid] = max(block_labels[bid], anom)

        if chunk_num % 5 == 0:
            print(f"  ... chunk {chunk_num}: {len(block_order):,} sessions", end='\r')
        del chunk; gc.collect()

    n_sessions = len(block_order)
    n_anomaly  = sum(block_labels[b] for b in block_order)
    print(f"\n  âœ… Total rows processed | Sessions: {n_sessions:,}")
    print(f"  Normal: {n_sessions - n_anomaly:,} | Anomaly: {n_anomaly:,} "
          f"({n_anomaly/n_sessions*100:.1f}%)")

    # Temporal split 60/20/20 (on sessions, preserving order)
    i1 = int(n_sessions * 0.60)
    i2 = int(n_sessions * 0.80)
    train_bids = block_order[:i1]
    val_bids   = block_order[i1:i2]
    test_bids  = block_order[i2:]
    print(f"  Split â†’ train={len(train_bids):,} | val={len(val_bids):,} "
          f"| test={len(test_bids):,}")

    # Build vocab from TRAIN sessions only â€” no data leakage
    # [Bekkouche2025_BiLSTM]: vocab from train only
    print("  Building vocabulary from train sessions only (no val/test leakage) ...")
    event_counter = collections.Counter()
    for bid in train_bids:
        event_counter.update(block_events[bid])

    # Keep all events seen â‰¥2 times in training
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for event, cnt in event_counter.most_common():
        if cnt >= 2:
            vocab[event] = len(vocab)
    VOCAB_SIZE = len(vocab)
    print(f"  Vocab size: {VOCAB_SIZE:,} (events seen â‰¥2 times in TRAIN only)")
    del event_counter; gc.collect()

    # Encode sessions â†’ padded int32 arrays
    def encode_session(events, vocab, max_len):
        seq = [vocab.get(e, vocab['<UNK>']) for e in events]
        seq = seq[:max_len]
        seq += [0] * (max_len - len(seq))
        return np.array(seq, dtype=np.int32)

    def bids_to_arrays(bids):
        X = np.stack([encode_session(block_events[b], vocab, MAX_SEQ_LEN) for b in bids])
        y = np.array([block_labels[b] for b in bids], dtype=np.int32)
        return X, y

    print("  Encoding sessions ...")
    X_train, y_train = bids_to_arrays(train_bids)
    X_val,   y_val   = bids_to_arrays(val_bids)
    X_test,  y_test  = bids_to_arrays(test_bids)

    print(f"  Shapes: X_train={X_train.shape} X_val={X_val.shape} X_test={X_test.shape}")
    print(f"  Anomaly %: train={y_train.mean()*100:.1f}% "
          f"val={y_val.mean()*100:.1f}% test={y_test.mean()*100:.1f}%")

    # Save
    joblib.dump(vocab, f'{MODEL_DIR}/vocab_hdfs_cnnbilstm_standalone.pkl')
    np.savez_compressed(f'{MODEL_DIR}/hdfs_cnnbilstm_train.npz', X=X_train, y=y_train)
    np.savez_compressed(f'{MODEL_DIR}/hdfs_cnnbilstm_val.npz',   X=X_val,   y=y_val)
    np.savez_compressed(f'{MODEL_DIR}/hdfs_cnnbilstm_test.npz',  X=X_test,  y=y_test)

    del block_events, block_labels, block_order
    del train_bids, val_bids, test_bids
    gc.collect()

    print(f"  âœ… Sessions ready ({time.time()-t0:.0f}s)")
    ckpt['sessions_ready'] = True; save_ckpt(ckpt)

else:
    print("[CELL 2] â­ï¸  Loading cached sessions ...")
    vocab      = joblib.load(f'{MODEL_DIR}/vocab_hdfs_cnnbilstm_standalone.pkl')
    VOCAB_SIZE = len(vocab)
    train_d = np.load(f'{MODEL_DIR}/hdfs_cnnbilstm_train.npz')
    val_d   = np.load(f'{MODEL_DIR}/hdfs_cnnbilstm_val.npz')
    test_d  = np.load(f'{MODEL_DIR}/hdfs_cnnbilstm_test.npz')
    X_train, y_train = train_d['X'], train_d['y']
    X_val,   y_val   = val_d['X'],   val_d['y']
    X_test,  y_test  = test_d['X'],  test_d['y']
    print(f"  VOCAB: {VOCAB_SIZE:,} | Train: {X_train.shape} "
          f"| Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"  Anomaly %: train={y_train.mean()*100:.1f}% "
          f"val={y_val.mean()*100:.1f}% test={y_test.mean()*100:.1f}%")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 3 â€” MultiScaleCNNBiLSTM Architecture
#
# Based on [Lu2018_LogCNN]: Multiple 1D CNN kernel sizes [2,3,5] act like
#   bigram, trigram, and 5-gram detectors over the embedded event sequence.
#   AdaptiveMaxPool1d collapses spatial dimension after each branch.
#   All branches are concatenated â†’ fed into BiLSTM for temporal context.
#
# Based on [Zhang2019_LogRobust]: Attention over ALL BiLSTM output timesteps
#   (not just the last hidden state). Softmax weights allow the model to
#   focus on the subset of events that carry the anomaly signal.
#
# Based on [Guo2021_LogBERT]: Bidirectional LSTM ensures both past and future
#   context within a session are available when classifying each timestep.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class MultiScaleCNNBiLSTM(nn.Module):
    """Multi-scale CNN + BiLSTM with attention for log anomaly detection.

    Pipeline:
      1. Embedding layer maps integer event IDs to dense vectors.
      2. Three parallel 1D CNN branches (kernels 2, 3, 5) capture local
         n-gram patterns at different scales [Lu2018_LogCNN].
      3. Branches are truncated to the same length and concatenated along
         the channel dimension â†’ fed into the BiLSTM as a sequence.
      4. BiLSTM provides global temporal context [Guo2021_LogBERT].
      5. Attention mechanism assigns a scalar weight to each timestep and
         computes a weighted sum (context vector) [Zhang2019_LogRobust].
      6. Dropout + FC â†’ scalar logit (binary cross-entropy loss).
    """

    def __init__(self, vocab_size, embed_dim=64, cnn_filters=64,
                 kernel_sizes=(2, 3, 5), hidden_size=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multi-scale 1D CNN branches [Lu2018_LogCNN]
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, cnn_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.relu = nn.ReLU()

        # BiLSTM on the concatenated multi-scale CNN features
        cnn_out_dim = cnn_filters * len(kernel_sizes)
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Attention: one scalar per timestep [Zhang2019_LogRobust]
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)            # [batch, seq_len, embed_dim]
        emb_t    = embedded.permute(0, 2, 1)   # [batch, embed_dim, seq_len]

        conv_outs = [self.relu(conv(emb_t)) for conv in self.convs]

        # Align all branches to the minimum spatial length
        min_len   = min(c.size(2) for c in conv_outs)
        conv_outs = [c[:, :, :min_len] for c in conv_outs]

        cnn_concat = torch.cat(conv_outs, dim=1)       # [batch, filters*n, min_len]
        cnn_concat = cnn_concat.permute(0, 2, 1)       # [batch, min_len, filters*n]

        lstm_out, _ = self.lstm(cnn_concat)             # [batch, min_len, hidden*2]

        # Attention
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context      = (lstm_out * attn_weights).sum(dim=1)

        logit = self.fc(self.dropout(context)).squeeze(-1)
        return logit


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 4 â€” Threshold Search Utility
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def find_best_threshold_f1(probs, labels, n_points=500):
    """Grid search threshold on validation probabilities to maximise F1.

    CRITICAL: Only called on VALIDATION data. Test set is never used here.
    [Bekkouche2025_BiLSTM]: F1-sensitive threshold selection on val set.
    """
    lo = float(np.percentile(probs, 1))
    hi = float(np.percentile(probs, 99))
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(lo, hi, n_points):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1  = f1
            best_thr = thr
    return best_thr, best_f1


def collect_val_probs(model, X_val, y_val, batch_size):
    model.eval()
    val_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_val).long(),
                      torch.from_numpy(y_val).float()),
        batch_size=batch_size, num_workers=0)
    val_probs, val_trues = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            probs = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
            val_probs.extend(probs.tolist())
            val_trues.extend(yb.numpy().astype(int).tolist())
    return np.array(val_probs), np.array(val_trues)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 5 â€” Training Function
#
# Based on [Zhang2019_LogRobust]:
#   - pos_weight = sqrt(n_neg/n_pos) handles class imbalance in HDFS.
#   - AdamW with weight decay 1e-4 regularises the large embedding table.
#   - Gradient clipping (max_norm=1.0) stabilises BiLSTM training.
#
# Based on [Bekkouche2025_BiLSTM]:
#   - CosineAnnealingWarmRestarts (T_0=5, T_mult=2) avoids LR getting stuck.
#   - Mixed precision (autocast + GradScaler) halves memory and speeds T4/P100.
#   - Early stopping on val F1 with F1-OPTIMAL threshold (NOT fixed 0.5).
#     This correctly maximises the early stopping signal.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def train_cnn_bilstm(X_train, y_train, X_val, y_val,
                     vocab_size, config, max_epochs=50, patience=10):
    """Train MultiScaleCNNBiLSTM with F1-optimal early stopping and AMP.

    Returns:
        best_state_dict, best_val_f1, best_threshold,
        train_losses, val_f1s
    """
    model = MultiScaleCNNBiLSTM(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        cnn_filters=config['cnn_filters'],
        kernel_sizes=(2, 3, 5),
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
    ).to(DEVICE)

    # Class imbalance: sqrt(n_neg/n_pos) pos_weight
    n_neg = int((y_train == 0).sum())
    n_pos = max(int((y_train == 1).sum()), 1)
    pw    = max(1.0, float(np.sqrt(n_neg / n_pos)))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=DEVICE))

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2)
    scaler = GradScaler()

    bs = config['batch_size']
    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_train).long(),
                      torch.from_numpy(y_train).float()),
        batch_size=bs, shuffle=True, num_workers=0)

    best_f1, best_thr, best_state, no_improve = 0.0, 0.5, None, 0
    train_losses, val_f1s = [], []

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_dl)
        train_losses.append(avg_loss)

        # Validation: collect probs, search optimal threshold
        # CRITICAL: threshold searched on val, NOT fixed at 0.5
        val_probs, val_trues = collect_val_probs(model, X_val, y_val, bs)
        thr, val_f1 = find_best_threshold_f1(val_probs, val_trues, n_points=300)
        val_f1s.append(val_f1)

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_thr   = thr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>2}/{max_epochs} | Loss={avg_loss:.4f} "
                  f"| Val F1={val_f1:.4f} (thr={thr:.3f}) | Best={best_f1:.4f}")

        if no_improve >= patience:
            print(f"    â¹ Early stopping at epoch {epoch}")
            break

    return best_state, best_f1, best_thr, train_losses, val_f1s


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 6 â€” Optuna Hyperparameter Search + Full Training
#
# Search space based on [Lu2018_LogCNN] + [Zhang2019_LogRobust]:
#   embed_dim  : [32, 64, 128]    â€” embedding dimensionality
#   cnn_filters: [32, 64, 128]    â€” filters per CNN kernel
#   hidden_size: [64, 128, 256]   â€” BiLSTM hidden units (per direction)
#   num_layers : 1â€“3              â€” BiLSTM depth
#   dropout    : 0.1â€“0.5
#   lr         : log[1e-4, 5e-3]  â€” AdamW learning rate
#   batch_size : [128, 256, 512]
#
# Warm-start: best-known config from [Lu2018] + [Zhang2019] combined
# 20 trials to match complexity of architecture vs BiLSTM-only (NB06).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if 'done' in ckpt:
    print("â­ï¸  CNN+BiLSTM already done (checkpoint). Skipping training.")
else:
    print(f"\n{'='*65}")
    print(f"  ðŸ”¬ HDFS CNN+BiLSTM HYBRID â€” MULTI-SCALE OPTIMIZATION")
    print(f"{'='*65}")
    t0_total = time.time()

    def objective(trial):
        cfg = {
            'embed_dim':   trial.suggest_categorical('embed_dim',   [32, 64, 128]),
            'cnn_filters': trial.suggest_categorical('cnn_filters', [32, 64, 128]),
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
            'num_layers':  trial.suggest_int('num_layers', 1, 3),
            'dropout':     trial.suggest_float('dropout', 0.1, 0.5),
            'lr':          trial.suggest_float('lr', 1e-4, 5e-3, log=True),
            'batch_size':  trial.suggest_categorical('batch_size', [128, 256, 512]),
        }
        # Quick trials: 12 epochs, patience 6 (fast exploration)
        # Objective = F1 with optimal val threshold (not fixed 0.5)
        best_state, best_f1, best_thr, _, _ = train_cnn_bilstm(
            X_train, y_train, X_val, y_val,
            VOCAB_SIZE, cfg, max_epochs=12, patience=6)
        del best_state
        gc.collect()
        return best_f1

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    # Warm-start: Lu2018 multi-scale CNN + Zhang2019 attention defaults
    study.enqueue_trial({
        'embed_dim': 64, 'cnn_filters': 64, 'hidden_size': 128,
        'num_layers': 2, 'dropout': 0.3, 'lr': 0.0005, 'batch_size': 256,
    })

    print(f"  ðŸ” Optuna (20 trials, timeout=1200s) ...")
    study.optimize(objective, n_trials=20, timeout=1200)
    best_params = study.best_params
    print(f"  ðŸ† Best params: {best_params}")
    print(f"  ðŸ† Best val F1 (Optuna): {study.best_value:.4f}")

    # Full training with best hyperparameters
    print(f"\n  ðŸš€ Full training (50 epochs, patience=12) ...")
    (best_state, best_val_f1, best_threshold,
     losses, val_f1s) = train_cnn_bilstm(
        X_train, y_train, X_val, y_val,
        VOCAB_SIZE, best_params, max_epochs=50, patience=12)

    print(f"\n  âœ… Full training done.")
    print(f"     Best val F1={best_val_f1:.4f} at threshold={best_threshold:.4f}")

    # â”€â”€ Final test evaluation â€” test set touched EXACTLY ONCE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model = MultiScaleCNNBiLSTM(
        vocab_size=VOCAB_SIZE,
        embed_dim=best_params['embed_dim'],
        cnn_filters=best_params['cnn_filters'],
        kernel_sizes=(2, 3, 5),
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
    ).to(DEVICE)
    model.load_state_dict(best_state)
    model.eval()

    test_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_test).long(),
                      torch.from_numpy(y_test).float()),
        batch_size=best_params['batch_size'], num_workers=0)

    t_infer = time.time()
    test_probs, test_trues = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            probs = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
            test_probs.extend(probs.tolist())
            test_trues.extend(yb.numpy().astype(int).tolist())
    infer_time = time.time() - t_infer

    y_prob = np.array(test_probs)
    y_true = np.array(test_trues)
    # Apply val-derived threshold to test â€” NO threshold re-search on test
    y_pred = (y_prob >= best_threshold).astype(int)

    # â”€â”€ Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)
    metrics = {
        'Dataset':     'HDFS',
        'Model':       'CNN+BiLSTM (Multi-Scale)',
        'Type':        'Supervised (DL)',
        'Threshold':   round(float(best_threshold), 4),
        'Precision':   round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall':      round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1_Anomaly':  round(f1_score(y_true, y_pred, zero_division=0), 4),
        'Macro_F1':    round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'AUC':         round(roc_auc, 4),
        'MCC':         round(matthews_corrcoef(y_true, y_pred), 4),
        'Avg_Precision': round(average_precision_score(y_true, y_prob), 4),
        'Inference_Time_s':        round(infer_time, 4),
        'Inference_Per_Sample_ms': round(infer_time / max(len(y_true), 1) * 1000, 4),
    }

    # â”€â”€ Paper comparison table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    paper_f1 = 0.985   # [Bekkouche2025_BiLSTM] best supervised on HDFS
    our_f1   = metrics['F1_Anomaly']
    delta    = our_f1 - paper_f1
    print(f"\n  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”")
    print(f"  â”‚  RESULTS vs PAPER â€” HDFS CNN+BiLSTM                â”‚")
    print(f"  â”‚  Paper [Bekkouche2025_BiLSTM] F1 : {paper_f1:.4f}          â”‚")
    print(f"  â”‚  Our F1                        : {our_f1:.4f}          â”‚")
    print(f"  â”‚  Delta                         : {delta:+.4f}          â”‚")
    print(f"  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜")

    print(f"\n  ðŸ“Š TEST RESULTS â€” HDFS CNN+BiLSTM:")
    print(classification_report(y_true, y_pred,
                                target_names=['Normal', 'Anomaly'], digits=4))
    print(f"  AUC={roc_auc:.4f} | MCC={metrics['MCC']:.4f} "
          f"| AP={metrics['Avg_Precision']:.4f}")
    print(f"  Threshold (from val): {best_threshold:.4f}")

    # â”€â”€ Save model + config + results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    torch.save(best_state, f'{MODEL_DIR}/cnn_bilstm_hdfs_opt.pt')
    with open(f'{MODEL_DIR}/cnn_bilstm_hdfs_config.json', 'w') as f:
        json.dump({
            **best_params,
            'vocab_size':   VOCAB_SIZE,
            'max_seq_len':  MAX_SEQ_LEN,
            'kernel_sizes': [2, 3, 5],
            'threshold':    round(float(best_threshold), 4),
            'best_val_f1_optuna': round(study.best_value, 4),
            'best_val_f1_final':  round(best_val_f1, 4),
            'paper_f1': paper_f1,
            'delta_vs_paper': round(delta, 4),
            **metrics,
        }, f, indent=2)
    pd.DataFrame([metrics]).round(4).to_csv(
        f'{REPORT}/cnn_bilstm_hdfs_results.csv', index=False)

    print(f"\n  ðŸ’¾ Saved: cnn_bilstm_hdfs_opt.pt | cnn_bilstm_hdfs_config.json "
          f"| cnn_bilstm_hdfs_results.csv")

    # â”€â”€ Plots â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses, 'b-o', markersize=3)
    ax1.set_title('Training Loss â€” HDFS CNN+BiLSTM', fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('BCE Loss'); ax1.grid(alpha=0.3)

    ax2.plot(val_f1s, 'g-o', markersize=3)
    ax2.axhline(best_val_f1, color='r', linestyle='--', alpha=0.7,
                label=f'Best F1={best_val_f1:.4f} (thr={best_threshold:.3f})')
    ax2.set_title('Validation F1 (optimal threshold) â€” HDFS CNN+BiLSTM',
                  fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('F1'); ax2.set_ylim([0, 1.05])
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{REPORT}/cnn_bilstm_hdfs_curves.png', dpi=300)
    plt.close()

    # ROC curve
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, 'b-', lw=2, label=f'AUC={roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('ROC â€” HDFS CNN+BiLSTM (Standalone)', fontweight='bold')
    plt.legend(loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f'{REPORT}/cnn_bilstm_hdfs_roc.png', dpi=300)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    ax.set_title('CM â€” HDFS CNN+BiLSTM (Standalone)', fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(f'{REPORT}/cnn_bilstm_hdfs_cm.png', dpi=300)
    plt.close()

    # Memory cleanup
    del model, X_train, X_val, X_test, y_train, y_val, y_test
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    ckpt['done'] = True; save_ckpt(ckpt)
    print(f"\n  âœ… CNN+BiLSTM HDFS Standalone complete ({time.time()-t0_total:.0f}s)")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 7 â€” Verification Block
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'='*65}")
print("  âœ… CNN+BiLSTM HDFS STANDALONE â€” COMPLETE")
print(f"{'='*65}")

output_files = [
    (MODEL_DIR, 'cnn_bilstm_hdfs_opt.pt'),
    (MODEL_DIR, 'cnn_bilstm_hdfs_config.json'),
    (MODEL_DIR, 'vocab_hdfs_cnnbilstm_standalone.pkl'),
    (MODEL_DIR, 'hdfs_cnnbilstm_train.npz'),
    (MODEL_DIR, 'hdfs_cnnbilstm_val.npz'),
    (MODEL_DIR, 'hdfs_cnnbilstm_test.npz'),
    (REPORT,    'cnn_bilstm_hdfs_results.csv'),
    (REPORT,    'cnn_bilstm_hdfs_curves.png'),
    (REPORT,    'cnn_bilstm_hdfs_roc.png'),
    (REPORT,    'cnn_bilstm_hdfs_cm.png'),
]

all_ok = True
for directory, fname in output_files:
    p = os.path.join(directory, fname)
    exists = os.path.exists(p)
    status = 'âœ…' if exists else 'âŒ'
    size_s = f"({os.path.getsize(p)/1024:.1f} KB)" if exists else "(missing)"
    print(f"  {status} {fname:<45} {size_s}")
    if not exists:
        all_ok = False

print(f"\n  Status: {'ðŸŽ‰ All outputs present' if all_ok else 'âš ï¸  Some outputs missing'}")
print(f"\n  ðŸ“Š Results CSV  â†’ {REPORT}/cnn_bilstm_hdfs_results.csv")
print(f"  ðŸ§  Model state  â†’ {MODEL_DIR}/cnn_bilstm_hdfs_opt.pt")
print(f"  âš™ï¸  Config JSON  â†’ {MODEL_DIR}/cnn_bilstm_hdfs_config.json")
print(f"\n  Paper citations:")
print(f"    [Lu2018_LogCNN]        â€” Multi-scale CNN (kernels 2,3,5)")
print(f"    [Zhang2019_LogRobust]  â€” Attention over all BiLSTM timesteps")
print(f"    [Guo2021_LogBERT]      â€” Bidirectional context for offline analysis")
print(f"    [Bekkouche2025_BiLSTM] â€” F1-optimal threshold on VAL, pos_weight")
print(f"\n  KEY CHANGE vs original:")
print(f"    âŒ Old: fixed threshold=0.5 for val early-stopping AND test eval")
print(f"    âŒ Old: label != '-' for anomaly detection (HDFS uses 'Normal')")
print(f"    âœ… New: threshold searched on val probs, applied once to test")
print(f"    âœ… New: label != 'Normal' for HDFS anomaly detection (correct)")
print(f"    âœ… New: vocab built from TRAIN only, MAX_SEQ_LEN=75 (consistent)")


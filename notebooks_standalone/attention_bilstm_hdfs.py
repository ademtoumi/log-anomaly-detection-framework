# =============================================================================
# STANDALONE NOTEBOOK 06 â€” HDFS Attention-BiLSTM (Fully Independent)
#
# âœ… ZERO dependencies â€” reads HDFS_Drain.csv directly from
#    /kaggle/input/pfe-log-anomaly/ and builds all sessions inline.
# âœ… One dataset only (HDFS) â€” RAM stays safe on Kaggle GPU.
# âœ… No output files from any other notebook are required.
#
# Architecture: Embedding â†’ BiLSTM â†’ Attention â†’ Dropout â†’ FC â†’ Sigmoid
#   Based on [Zhang2019_LogRobust]: Attention over ALL hidden states is
#   superior to using only the last hidden state for log anomaly detection.
#
# Session grouping:
#   Based on [Du2017_DeepLog]: HDFS sessions are naturally grouped by
#   BlockId extracted from the raw log lines.
#
# Bidirectional context:
#   Based on [Guo2021_LogBERT]: Bidirectional context is superior for
#   offline (post-hoc) log analysis tasks.
#
# Threshold & evaluation:
#   Based on [Bekkouche2025_BiLSTM]: F1-sensitive threshold selection on
#   VALIDATION SET ONLY. Test set touched exactly once.
#   Fixed threshold = wrong result. Threshold searched on val probabilities.
#
# Kaggle setup:
#   - Add dataset: pfe-log-anomaly  (contains HDFS_Drain.csv)
#   - Accelerator: GPU T4 or P100
#   - Estimated time: ~35 minutes
#
# Outputs saved to pfe_report/ and models/:
#   - hdfs_sessions_{train,val,test}.npz
#   - vocab_hdfs_opt.pkl
#   - bilstm_hdfs_opt.pt
#   - bilstm_hdfs_config.json
#   - bilstm_hdfs_results.csv
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
    classification_report, confusion_matrix, roc_curve, auc,
    f1_score, precision_score, recall_score, matthews_corrcoef,
    average_precision_score,
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# â”€â”€ Fixed seeds everywhere â€” reproducibility [Bekkouche2025_BiLSTM] â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 1 â€” Environment & Paths
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
KAGGLE    = os.path.exists('/kaggle')
BASE_IN   = '/kaggle/input/pfe-log-anomaly' if KAGGLE else 'Dataset'
BASE_OUT  = '/kaggle/working'               if KAGGLE else 'result/results_attention_bilstm_hdfs'
MODEL_DIR = f'{BASE_OUT}/models'
REPORT    = f'{BASE_OUT}/pfe_report'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT, exist_ok=True)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"âœ… Device: {DEVICE} | {'Kaggle' if KAGGLE else 'Local'} environment")

# Checkpoint system â€” re-running skips completed steps
CKPT = pathlib.Path(BASE_OUT) / 'ckpt_06_bilstm_hdfs_standalone.json'

def save_ckpt(d):
    with open(CKPT, 'w') as f:
        json.dump(d, f)

def load_ckpt():
    if CKPT.exists():
        with open(CKPT) as f:
            return json.load(f)
    return {}

ckpt = load_ckpt()
print(f"  Checkpoint keys loaded: {list(ckpt.keys())}")

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
MAX_SEQ_LEN = 75   # [Du2017]: most HDFS sessions < 50 events; margin added

if 'sessions_ready' not in ckpt:
    print(f"\n{'='*65}")
    print("  [CELL 2] Building HDFS Sessions from HDFS_Drain.csv ...")
    print(f"{'='*65}")
    t0 = time.time()

    filepath = find_file('HDFS_Drain.csv')
    print(f"  Reading: {filepath}")

    # â”€â”€ Phase 1: Chunked session aggregation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # block_events: BlockId â†’ list of template strings (in arrival order)
    # block_labels: BlockId â†’ 1 if ANY log line for that block is anomalous
    # block_order:  insertion-ordered list of BlockIds (preserves temporal order)
    block_events = {}
    block_labels = {}
    block_order  = []

    chunk_num = 0
    for chunk in pd.read_csv(filepath, chunksize=500_000,
                              on_bad_lines='skip', low_memory=False):
        chunk_num += 1

        # Extract BlockId â€” prefer 'BlockId' column, else regex from 'log'
        if 'BlockId' in chunk.columns:
            chunk['_bid'] = chunk['BlockId'].astype(str).str.strip()
        else:
            chunk['_bid'] = chunk['log'].str.extract(r'(blk_-?\d+)')

        chunk = chunk.dropna(subset=['_bid'])

        # Anomaly label â€” 'Label' column with 'Normal' vs anything else
        # [Du2017_DeepLog]: HDFS ground truth uses 'Normal' / 'Anomaly' labels
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
            # Session label = OR across all its lines (any anomaly â†’ anomaly)
            block_labels[bid] = max(block_labels[bid], int(row['_anom']))

        if chunk_num % 5 == 0:
            print(f"     Chunk {chunk_num}: {len(block_events):,} unique blocks so far")
        del chunk
        gc.collect()

    n_blocks = len(block_order)
    n_anomaly_blocks = sum(block_labels.values())
    print(f"\n  Total blocks : {n_blocks:,}")
    print(f"  Anomaly blocks: {n_anomaly_blocks:,} ({n_anomaly_blocks/n_blocks*100:.1f}%)")

    # â”€â”€ Phase 2: Build vocabulary from TRAIN sessions only â€” no leakage â”€â”€â”€â”€â”€â”€
    # [Bekkouche2025_BiLSTM]: vocab from train only prevents test-set leakage
    i1_vocab = int(n_blocks * 0.60)
    train_bids = block_order[:i1_vocab]

    all_templates = set()
    for bid in train_bids:
        all_templates.update(block_events[bid])

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for idx, tmpl in enumerate(sorted(all_templates)):
        vocab[tmpl] = idx + 2
    VOCAB_SIZE = len(vocab)
    joblib.dump(vocab, f'{MODEL_DIR}/vocab_hdfs_opt.pkl')
    print(f"  Vocabulary: {VOCAB_SIZE} unique templates (+2 special tokens, train-only)")

    # â”€â”€ Phase 3: Encode sessions as padded int32 sequences â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    sequences  = np.zeros((n_blocks, MAX_SEQ_LEN), dtype=np.int32)
    labels_arr = np.zeros(n_blocks, dtype=np.int32)

    for i, bid in enumerate(block_order):
        events    = block_events[bid]
        event_ids = [vocab.get(e, 1) for e in events]   # <UNK>=1 fallback
        seq_len   = min(len(event_ids), MAX_SEQ_LEN)
        sequences[i, :seq_len] = event_ids[:seq_len]
        labels_arr[i] = block_labels[bid]

    # Free the large dicts now â€” we only need sequences & labels_arr
    del block_events, block_labels
    gc.collect()
    print(f"  Sequences shape: {sequences.shape} | dtype: {sequences.dtype}")

    # â”€â”€ Phase 4: Temporal split 60 / 20 / 20 on blocks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    i1 = int(n_blocks * 0.60)
    i2 = int(n_blocks * 0.80)

    np.savez_compressed(f'{MODEL_DIR}/hdfs_sessions_train.npz',
                        X=sequences[:i1],  y=labels_arr[:i1])
    np.savez_compressed(f'{MODEL_DIR}/hdfs_sessions_val.npz',
                        X=sequences[i1:i2], y=labels_arr[i1:i2])
    np.savez_compressed(f'{MODEL_DIR}/hdfs_sessions_test.npz',
                        X=sequences[i2:],  y=labels_arr[i2:])

    print(f"\n  Train : {i1:,} blocks | anomaly={labels_arr[:i1].mean()*100:.1f}%")
    print(f"  Val   : {i2-i1:,} blocks | anomaly={labels_arr[i1:i2].mean()*100:.1f}%")
    print(f"  Test  : {n_blocks-i2:,} blocks | anomaly={labels_arr[i2:].mean()*100:.1f}%")

    del sequences, labels_arr
    gc.collect()

    elapsed = time.time() - t0
    ckpt['sessions_ready'] = True
    save_ckpt(ckpt)
    print(f"\n  âœ… Sessions saved in {elapsed:.0f}s")
    print(f"     â†’ {MODEL_DIR}/hdfs_sessions_{{train,val,test}}.npz")
    print(f"     â†’ {MODEL_DIR}/vocab_hdfs_opt.pkl")
else:
    print("[CELL 2] â­ï¸  Sessions already built (checkpoint 'sessions_ready')")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 3 â€” Attention-BiLSTM Architecture
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Based on [Zhang2019_LogRobust]:
#   Attention over ALL hidden states > using only the last hidden state.
#   This allows the model to focus on the most anomalous events in a sequence.
#
# Based on [Guo2021_LogBERT]:
#   Bidirectional context is superior for offline (batch) log analysis.

class AttentionBiLSTM(nn.Module):
    """BiLSTM with attention mechanism for HDFS session-level anomaly detection.

    Key design choices:
    - Bidirectional LSTM captures both forward and backward context in a session.
    - Attention (softmax over all timestep outputs) lets the model focus on the
      most anomalous events rather than relying solely on the final hidden state.
    - padding_idx=0 ensures <PAD> tokens do not contribute to gradients.

    References:
        [Zhang2019_LogRobust] â€” Attention mechanism for log anomaly detection.
        [Guo2021_LogBERT]     â€” Bidirectional context is best for offline analysis.
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_size=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        # Attention: scalar score per timestep â†’ softmax â†’ weighted sum
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # x: [batch, seq_len]  (int token ids)
        embedded  = self.embedding(x)                        # [B, T, embed_dim]
        lstm_out, _ = self.lstm(embedded)                    # [B, T, hidden*2]

        # Attention over all T timesteps â€” [Zhang2019_LogRobust]
        attn_scores  = self.attention(lstm_out)              # [B, T, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)    # [B, T, 1]
        context      = (lstm_out * attn_weights).sum(dim=1) # [B, hidden*2]

        dropped = self.dropout(context)
        logits  = self.fc(dropped).squeeze(-1)              # [B]
        return logits


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 4 â€” Training Function
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Training design:
#   - Mixed precision (autocast + GradScaler) for GPU memory efficiency.
#   - pos_weight = sqrt(n_neg / n_pos) to handle class imbalance.
#     Based on [Bekkouche2025_BiLSTM]: sqrt damping avoids over-penalizing normals.
#   - CosineAnnealingWarmRestarts (T_0=5, T_mult=2) to escape local minima.
#   - Early stopping (patience=10) on val F1 with F1-OPTIMAL threshold.
#     CRITICAL: val F1 is computed with the optimal threshold from val probabilities,
#     not fixed 0.5. This maximises the signal for early stopping.
#   - Gradient clipping (max_norm=1.0) for LSTM training stability.

def find_best_threshold_f1(probs, labels, n_points=500):
    """Grid search threshold on validation probabilities to maximise F1.
    
    Based on [Bekkouche2025_BiLSTM]: F1-sensitive threshold selection.
    Threshold is searched exclusively on the VALIDATION set.
    Test set is never used in threshold selection.
    
    Args:
        probs:    numpy array of predicted probabilities (sigmoid outputs)
        labels:   numpy array of true binary labels
        n_points: number of candidate threshold values to search
    Returns:
        (best_threshold, best_f1)
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
    """Collect validation probabilities and find optimal threshold."""
    model.eval()
    val_ds = TensorDataset(
        torch.from_numpy(X_val).long(),
        torch.from_numpy(y_val).float())
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_probs, val_trues = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            probs = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
            val_probs.extend(probs.tolist())
            val_trues.extend(yb.numpy().astype(int).tolist())
    return np.array(val_probs), np.array(val_trues)


def train_bilstm(X_train, y_train, X_val, y_val,
                 vocab_size, config, max_epochs=50, patience=10):
    """Train AttentionBiLSTM with early stopping and mixed precision.

    Early stopping criterion: val F1 computed with F1-OPTIMAL threshold
    (not fixed 0.5). This correctly maximises detection quality.

    Args:
        X_train/X_val: int32 numpy arrays of shape (N, MAX_SEQ_LEN)
        y_train/y_val: int32 numpy arrays of shape (N,)
        vocab_size: total vocabulary size
        config: dict with embed_dim, hidden_size, num_layers, dropout, lr, batch_size
        max_epochs: maximum training epochs
        patience: early stopping patience on val F1

    Returns:
        (best_state_dict, best_val_f1, best_threshold,
         train_losses, val_f1s)
    """
    model = AttentionBiLSTM(
        vocab_size   = vocab_size,
        embed_dim    = config['embed_dim'],
        hidden_size  = config['hidden_size'],
        num_layers   = config['num_layers'],
        dropout      = config['dropout'],
    ).to(DEVICE)

    # Class imbalance: sqrt(n_neg/n_pos) pos_weight
    # [Bekkouche2025_BiLSTM]: sqrt dampening is less aggressive than direct ratio
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pw    = max(1.0, float(np.sqrt(n_neg / max(n_pos, 1))))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], device=DEVICE))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=1e-4)
    # CosineAnnealingWarmRestarts: T_0=5 epochs, doubling period each restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2)
    scaler = GradScaler()

    bs = config['batch_size']
    train_ds = TensorDataset(
        torch.from_numpy(X_train).long(),
        torch.from_numpy(y_train).float())
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0)

    best_f1, best_thr, best_state = 0.0, 0.5, None
    no_improve = 0
    train_losses, val_f1s = [], []

    for epoch in range(1, max_epochs + 1):
        # â”€â”€ Training pass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                logits = model(xb)
                loss   = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        scheduler.step()

        avg_loss = epoch_loss / len(train_dl)
        train_losses.append(avg_loss)

        # â”€â”€ Validation: collect probs, search optimal threshold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # CRITICAL: threshold is searched on val set, NOT fixed at 0.5
        # [Bekkouche2025_BiLSTM]: F1-optimal threshold on validation data
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
            print(f"    Epoch {epoch:>2}/{max_epochs} | "
                  f"Loss={avg_loss:.4f} | Val F1={val_f1:.4f} (thr={thr:.3f}) | Best={best_f1:.4f}")

        if no_improve >= patience:
            print(f"    â¹ Early stopping at epoch {epoch} (patience={patience})")
            break

    return best_state, best_f1, best_thr, train_losses, val_f1s


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 5 â€” Optuna Hyperparameter Search + Full Training
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if 'done' in ckpt:
    print("â­ï¸  [CELL 5] BiLSTM training already done (checkpoint 'done')")
else:
    print(f"\n{'='*65}")
    print("  ðŸ§  HDFS ATTENTION-BiLSTM â€” OPTUNA + FULL TRAINING")
    print(f"{'='*65}")

    t0_total = time.time()

    # â”€â”€ Load sessions saved in CELL 2 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n  Loading sessions from models/ ...")
    train_data = np.load(f'{MODEL_DIR}/hdfs_sessions_train.npz')
    val_data   = np.load(f'{MODEL_DIR}/hdfs_sessions_val.npz')
    test_data  = np.load(f'{MODEL_DIR}/hdfs_sessions_test.npz')
    vocab      = joblib.load(f'{MODEL_DIR}/vocab_hdfs_opt.pkl')
    VOCAB_SIZE = len(vocab)

    X_train = train_data['X']   # int32 [N_train, MAX_SEQ_LEN]
    y_train = train_data['y']   # int32 [N_train]
    X_val   = val_data['X']
    y_val   = val_data['y']
    X_test  = test_data['X']
    y_test  = test_data['y']

    print(f"  VOCAB: {VOCAB_SIZE} | MAX_SEQ_LEN: {MAX_SEQ_LEN}")
    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"  Anomaly rate â†’ train={y_train.mean()*100:.1f}% "
          f"val={y_val.mean()*100:.1f}% test={y_test.mean()*100:.1f}%")

    # â”€â”€ Optuna: 20 trials, timeout=1200s â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # CRITICAL: Optuna objective uses F1-OPTIMAL threshold on val set.
    # Each trial runs max_epochs=12 with patience=6 (fast exploration mode).
    # Warm-start with literature-grounded defaults:
    #   [Zhang2019_LogRobust]: embed=64, hidden=128, layers=2, dropout=0.3
    #   [Du2017_DeepLog]: lrâ‰ˆ5e-4, batch=256

    def objective(trial):
        cfg = {
            'embed_dim':   trial.suggest_categorical('embed_dim',   [32, 64, 128]),
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
            'num_layers':  trial.suggest_int('num_layers', 1, 3),
            'dropout':     trial.suggest_float('dropout', 0.1, 0.5),
            'lr':          trial.suggest_float('lr', 1e-4, 5e-3, log=True),
            'batch_size':  trial.suggest_categorical('batch_size', [128, 256, 512]),
        }
        best_state, best_f1, best_thr, _, _ = train_bilstm(
            X_train, y_train, X_val, y_val,
            VOCAB_SIZE, cfg, max_epochs=12, patience=6)
        del best_state
        gc.collect()
        return best_f1   # F1 with optimal val threshold

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    # Warm-start trial from paper defaults â€” [Zhang2019_LogRobust] + [Du2017_DeepLog]
    study.enqueue_trial({
        'embed_dim': 64, 'hidden_size': 128, 'num_layers': 2,
        'dropout': 0.3, 'lr': 0.0005, 'batch_size': 256,
    })

    print(f"\n  ðŸ” Optuna search (20 trials, timeout=1200s) ...")
    study.optimize(objective, n_trials=20, timeout=1200)
    best_params = study.best_params
    print(f"  ðŸ† Best params : {best_params}")
    print(f"  ðŸ† Best val F1 (Optuna): {study.best_value:.4f}")

    # â”€â”€ Full training with best hyperparameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n  ðŸš€ Full training (max_epochs=50, patience=12) ...")
    (best_state, best_val_f1, best_threshold,
     train_losses, val_f1s) = train_bilstm(
        X_train, y_train, X_val, y_val,
        VOCAB_SIZE, best_params, max_epochs=50, patience=12)

    print(f"\n  âœ… Full training done.")
    print(f"     Best val F1={best_val_f1:.4f} at threshold={best_threshold:.4f}")

    # â”€â”€ Final model reload + TEST evaluation (test set touched exactly once) â”€â”€
    model = AttentionBiLSTM(
        vocab_size   = VOCAB_SIZE,
        embed_dim    = best_params['embed_dim'],
        hidden_size  = best_params['hidden_size'],
        num_layers   = best_params['num_layers'],
        dropout      = best_params['dropout'],
    ).to(DEVICE)
    model.load_state_dict(best_state)
    model.eval()

    test_ds = TensorDataset(
        torch.from_numpy(X_test).long(),
        torch.from_numpy(y_test).float())
    test_dl = DataLoader(test_ds, batch_size=best_params['batch_size'],
                         shuffle=False, num_workers=0)

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
    # Apply val-derived threshold to test â€” no threshold re-search on test!
    y_pred = (y_prob >= best_threshold).astype(int)

    # â”€â”€ Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    metrics = {
        'Dataset':    'HDFS',
        'Model':      'Attention-BiLSTM',
        'Type':       'Supervised (DL)',
        'Threshold':  round(float(best_threshold), 4),
        'Precision':  round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        'Recall':     round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        'F1_Anomaly': round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        'Macro_F1':   round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'AUC':        round(roc_auc, 4),
        'MCC':        round(matthews_corrcoef(y_true, y_pred), 4),
        'Avg_Precision':          round(average_precision_score(y_true, y_prob), 4),
        'Inference_Time_s':       round(infer_time, 4),
        'Inference_Per_Sample_ms': round(infer_time / max(len(y_true), 1) * 1000, 4),
    }

    # â”€â”€ Paper comparison table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # [Bekkouche2025_BiLSTM] reports F1=0.983 on HDFS for BiLSTM supervised.
    paper_f1 = 0.983
    our_f1   = metrics['F1_Anomaly']
    delta    = our_f1 - paper_f1
    print(f"\n  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”")
    print(f"  â”‚  RESULTS vs PAPER â€” HDFS Attention-BiLSTM           â”‚")
    print(f"  â”‚  Paper [Bekkouche2025_BiLSTM] F1 : {paper_f1:.4f}          â”‚")
    print(f"  â”‚  Our F1                        : {our_f1:.4f}          â”‚")
    print(f"  â”‚  Delta                         : {delta:+.4f}          â”‚")
    print(f"  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜")

    print(f"\n  ðŸ“Š TEST RESULTS â€” HDFS Attention-BiLSTM:")
    print(classification_report(
        y_true, y_pred,
        target_names=['Normal', 'Anomaly'],
        digits=4))
    print(f"  AUC={roc_auc:.4f} | MCC={metrics['MCC']:.4f} | "
          f"AP={metrics['Avg_Precision']:.4f}")
    print(f"  Threshold (from val): {best_threshold:.4f}")
    print(f"  Inference: {infer_time:.3f}s total | "
          f"{metrics['Inference_Per_Sample_ms']:.4f}ms/sample")

    # â”€â”€ Save model & results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    torch.save(best_state, f'{MODEL_DIR}/bilstm_hdfs_opt.pt')
    print(f"  âœ… Model saved â†’ {MODEL_DIR}/bilstm_hdfs_opt.pt")

    config_out = {
        **best_params,
        'vocab_size':   VOCAB_SIZE,
        'max_seq_len':  MAX_SEQ_LEN,
        'best_val_f1':  round(best_val_f1, 4),
        'threshold':    round(float(best_threshold), 4),
        'paper_f1':     paper_f1,
        'delta_vs_paper': round(delta, 4),
        **{k: (round(v, 4) if isinstance(v, float) else v)
           for k, v in metrics.items()},
    }
    with open(f'{MODEL_DIR}/bilstm_hdfs_config.json', 'w') as f:
        json.dump(config_out, f, indent=2)
    print(f"  âœ… Config saved â†’ {MODEL_DIR}/bilstm_hdfs_config.json")

    pd.DataFrame([metrics]).round(4).to_csv(
        f'{REPORT}/bilstm_hdfs_results.csv', index=False)
    print(f"  âœ… Results saved â†’ {REPORT}/bilstm_hdfs_results.csv")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # CELL 6 â€” Plots
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n  ðŸ“ˆ Generating plots ...")

    # Training loss + Val F1 curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(range(1, len(train_losses)+1), train_losses, 'b-o', markersize=3)
    ax1.set_title('Training Loss â€” HDFS Attention-BiLSTM', fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('BCE Loss'); ax1.grid(alpha=0.3)

    ax2.plot(range(1, len(val_f1s)+1), val_f1s, 'g-o', markersize=3)
    ax2.axhline(best_val_f1, color='r', linestyle='--', alpha=0.6,
                label=f'Best F1={best_val_f1:.4f} (thr={best_threshold:.3f})')
    ax2.set_title('Validation F1 (optimal threshold) â€” HDFS Attention-BiLSTM',
                  fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('F1 (Anomaly class)')
    ax2.set_ylim([0, 1.05]); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{REPORT}/bilstm_hdfs_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  âœ… Curves â†’ {REPORT}/bilstm_hdfs_curves.png")

    # ROC Curve
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve â€” HDFS Attention-BiLSTM', fontweight='bold')
    plt.legend(loc='lower right'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{REPORT}/bilstm_hdfs_roc.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  âœ… ROC    â†’ {REPORT}/bilstm_hdfs_roc.png")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix â€” HDFS Attention-BiLSTM', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{REPORT}/bilstm_hdfs_cm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  âœ… CM     â†’ {REPORT}/bilstm_hdfs_cm.png")

    # â”€â”€ Cleanup GPU memory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    del model, X_train, X_val, X_test
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        print(f"  ðŸ§¹ GPU cache cleared")

    ckpt['done'] = True
    save_ckpt(ckpt)
    total_time = time.time() - t0_total
    print(f"\n  âœ… HDFS BiLSTM complete â€” total wall time: {total_time:.0f}s "
          f"({total_time/60:.1f} min)")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 7 â€” Verification Block
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'='*65}")
print("  âœ…  STANDALONE NOTEBOOK 06 â€” HDFS ATTENTION-BiLSTM â€” COMPLETE")
print(f"{'='*65}")

expected_files = [
    (MODEL_DIR, 'hdfs_sessions_train.npz'),
    (MODEL_DIR, 'hdfs_sessions_val.npz'),
    (MODEL_DIR, 'hdfs_sessions_test.npz'),
    (MODEL_DIR, 'vocab_hdfs_opt.pkl'),
    (MODEL_DIR, 'bilstm_hdfs_opt.pt'),
    (MODEL_DIR, 'bilstm_hdfs_config.json'),
    (REPORT,    'bilstm_hdfs_results.csv'),
    (REPORT,    'bilstm_hdfs_curves.png'),
    (REPORT,    'bilstm_hdfs_roc.png'),
    (REPORT,    'bilstm_hdfs_cm.png'),
]

print(f"\n  Output file status:")
all_ok = True
for directory, fname in expected_files:
    path   = os.path.join(directory, fname)
    exists = os.path.exists(path)
    icon   = 'âœ…' if exists else 'âŒ'
    size_s = f"({os.path.getsize(path)/1024:.1f} KB)" if exists else "(missing)"
    print(f"    {icon} {fname:<45} {size_s}")
    if not exists:
        all_ok = False

print(f"\n  Checkpoint keys: {list(ckpt.keys())}")
print(f"  Status: {'ðŸŽ‰ All outputs present' if all_ok else 'âš ï¸  Some outputs missing'}")
print(f"\n  Model dir  : {MODEL_DIR}")
print(f"  Report dir : {REPORT}")
print(f"\n  Paper citations in this notebook:")
print(f"    [Zhang2019_LogRobust]  â€” Attention over ALL hidden states > last hidden state")
print(f"    [Du2017_DeepLog]       â€” HDFS sessions grouped by BlockId")
print(f"    [Guo2021_LogBERT]      â€” Bidirectional context for offline analysis")
print(f"    [Bekkouche2025_BiLSTM] â€” F1-optimal threshold on VAL, pos_weight strategy")
print(f"\n  KEY CHANGE vs original:")
print(f"    âŒ Old: fixed threshold=0.5 for both val early-stopping AND test evaluation")
print(f"    âœ… New: threshold searched on val probs, applied once to test")


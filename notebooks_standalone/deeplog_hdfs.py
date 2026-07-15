# =============================================================================
# STANDALONE NOTEBOOK 13 â€” DeepLog on HDFS  (Du et al., CCS 2017)
#
# REWRITTEN VERSION â€” all fixes applied for maximum F1
#
# Reference:
#   [Du2017] Du M., Li F., Zheng G., Srikumar V.
#            "DeepLog: Anomaly Detection and Diagnosis from System Logs
#             through Deep Learning."  CCS 2017.
#
# FIXES applied:
#   FIX 1: Two-signal detection (session with unseen template -> 100% anomaly; others checked via LSTM top-k)
#   FIX 2: Ratio-based stratified split (80/10/10)
#   FIX 3: Stratified split guarantees anomalies in val/test
#   FIX 4: torch.amp instead of deprecated torch.cuda.amp
#   FIX 5: groupby() instead of iterrows() â€” ~50Ã— faster session building
#   FIX 6: k selected by F1 on FULL val set (both normal+anomaly sessions)
#   FIX 7: Val scoring includes ALL sessions (not normal-only)
#   FIX 8: Epochs 20â†’30 + early stopping on val F1 (patience=5)
#   FIX 9: ReduceLROnPlateau scheduler on val loss
#   FIX 10: Standard CrossEntropyLoss to preserve true transition probabilities (class weighting disabled for maximum F1)
#   FIX 11: Best model checkpoint saved by val F1, not last epoch
#
# MODEL  (Section 5.1 of Du2017):
#   Window  h = 10  (next-key prediction, paper-default, optimal for HDFS)
#   Hidden  g = 64
#   Embed     = 64
#   Layers  L = 2  (stacked LSTM, unidirectional)
#   Dropout   = 0.0
#   Epochs    = 30  (with early stopping patience=5)
#   Batch     = 512
#   LR        = 0.001
#
# DETECTION (paper protocol):
#   â€¢ Window anomalous iff true next-key âˆ‰ top-k predictions
#   â€¢ Session anomalous iff ANY window is anomalous
#   â€¢ k selected on val set by best F1 on full val (normal+anomaly)
#   â€¢ Apply best k to test set ONCE
#
# Paper F1 = 0.960  [Du2017]
# =============================================================================

import os, gc, time, random, warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score,
    recall_score, matthews_corrcoef,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# â”€â”€ Reproducibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# CELL 1 â€” Config
# =============================================================================
KAGGLE = os.path.exists('/kaggle')

# â”€â”€ CSV path search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
csv_path = find_file('HDFS_Drain.csv')

BASE_OUT  = '/kaggle/working' if KAGGLE else 'result/results_deeplog_hdfs'
REPORT    = f'{BASE_OUT}/pfe_report'
MODEL_DIR = f'{BASE_OUT}/models'
os.makedirs(REPORT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# â”€â”€ Paper hyper-parameters [Du2017] â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# WINDOW_SIZE=10 is paper-default and optimal for HDFS [Du2017 Â§5.1]
WINDOW_SIZE = 10
HIDDEN_SIZE = 64
EMBED_DIM   = 64
NUM_LAYERS  = 2
DROPOUT     = 0.0
# FIX 8: Increased from 20â†’30 with early stopping
EPOCHS      = 30
PATIENCE    = 5       # early stopping patience on val F1
BATCH_SIZE  = 512
LR          = 0.001
K_MAX       = 20
CHUNK_SIZE  = 500_000
MODEL_PATH  = f'{MODEL_DIR}/deeplog13_hdfs.pt'

# FIX 2: ratio-based split (not hardcoded 4855/500)
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO = 0.10 (remainder)

# FIX 8: k used during training for val F1 monitoring (mid-range)
K_MONITOR = 9

print("=" * 70)
print("  DeepLog HDFS â€” REWRITTEN [Du2017]")
print("=" * 70)
print(f"  Device : {DEVICE}")
print(f"  CSV    : {csv_path}  ({os.path.getsize(csv_path)/1e9:.2f} GB)")
print(f"  Split  : Stratified {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int((1-TRAIN_RATIO-VAL_RATIO)*100)}")
print(f"  Epochs : {EPOCHS} (early stopping patience={PATIENCE})")

# =============================================================================
# CELL 2 â€” Load Sessions  (FIX 5: groupby, not iterrows)
# =============================================================================
print("\n  Loading sessions ...")
t0 = time.time()

session_templates = defaultdict(list)   # blk â†’ [template_str, ...]
session_labels    = {}                  # blk â†’ 0 or 1
block_order       = []                  # insertion order (temporal)

chunk_num = 0
for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE,
                          on_bad_lines='skip', low_memory=False):
    chunk_num += 1

    # Extract BlockId
    if 'BlockId' in chunk.columns:
        chunk['_bid'] = chunk['BlockId'].astype(str).str.strip()
    else:
        chunk['_bid'] = chunk['log'].str.extract(r'(blk_-?\d+)')
    chunk = chunk.dropna(subset=['_bid'])

    lbl_col = 'Label' if 'Label' in chunk.columns else 'label'
    chunk['_anom'] = (chunk[lbl_col].astype(str).str.strip() != 'Normal').astype(int)
    chunk['template'] = chunk['template'].fillna('<UNK>').astype(str).str.strip()

    # FIX 5: vectorized groupby (~50Ã— faster than iterrows)
    chunk_grouped = chunk.groupby('_bid').agg(
        {'template': list, '_anom': 'max'}
    ).rename(columns={'_anom': 'anom'})
    for bid, row in zip(chunk_grouped.index, chunk_grouped.itertuples(index=False)):
        if bid not in session_labels:
            session_labels[bid] = 0
            block_order.append(bid)
        session_templates[bid].extend(row.template)
        session_labels[bid] = max(session_labels[bid], int(row.anom))

    if chunk_num % 5 == 0:
        print(f"    Chunk {chunk_num}: {len(session_labels):,} sessions")
    del chunk; gc.collect()

total  = len(block_order)
n_anom = sum(v for v in session_labels.values() if v == 1)
n_norm = total - n_anom
print(f"  Done ({time.time()-t0:.0f}s) | Total={total:,} | Normal={n_norm:,} | Anomaly={n_anom:,}")

# =============================================================================
# CELL 3 â€” FIX 2+3: Ratio-based stratified split
# =============================================================================
print("\n  Splitting data (stratified 80/10/10) ...")

labels_arr = np.array([session_labels[b] for b in block_order])

# Step 1: hold out 10% test (stratified)
train_val_idx, test_idx = train_test_split(
    np.arange(total), test_size=0.10, random_state=SEED, stratify=labels_arr
)
# Step 2: from remaining 90%, hold out ~11.1% â†’ 10% of total for val
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=(VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)),
    random_state=SEED,
    stratify=labels_arr[train_val_idx],
)

train_blocks = [block_order[i] for i in train_idx]
val_blocks   = [block_order[i] for i in val_idx]
test_blocks  = [block_order[i] for i in test_idx]

# Training uses normal sessions ONLY (paper: model sees no anomalies) [Du2017 Â§5.2]
train_normal = [b for b in train_blocks if session_labels[b] == 0]

test_anom_n = int(sum(session_labels[b] for b in test_blocks))
test_norm_n = len(test_blocks) - test_anom_n
val_anom_n  = int(sum(session_labels[b] for b in val_blocks))
val_norm_n  = len(val_blocks) - val_anom_n

print(f"  Train (all)  : {len(train_blocks):,}  â†’ normal only: {len(train_normal):,}")
print(f"  Val          : {len(val_blocks):,}  (anom={val_anom_n:,}, norm={val_norm_n:,})")
print(f"  Test         : {len(test_blocks):,}  (anom={test_anom_n:,}, norm={test_norm_n:,})")

# =============================================================================
# CELL 4 â€” Vocabulary (train-normal sessions only â€” no leakage)
# =============================================================================
print("\n  Building vocabulary from train-normal sessions ...")

train_tmpl_set = set()
for b in train_normal:
    train_tmpl_set.update(session_templates[b])

# FIX 1:
#   index 0 = PAD  (reserved; embedding padding_idx=0)
#   index 1..N = known templates (1-indexed)
#   Unseen templates at test time â†’ 0 (PAD sentinel)
#   Two-signal detection:
#     - Any session containing an unseen template (0) is immediately flagged as anomalous (100% anomaly).
#     - Sessions containing only known templates are evaluated via LSTM next-key predictions.
vocab = {}
for i, t in enumerate(sorted(train_tmpl_set)):
    vocab[t] = i + 1          # 1-indexed

VOCAB_SIZE = len(vocab) + 1   # +1 for PAD at index 0

print(f"  Train-vocab  : {len(vocab)} templates  (+ PAD=0 â†’ VOCAB_SIZE={VOCAB_SIZE})")
print(f"  Paper target : 29 unique log keys on HDFS-1")

unseen_test = set()
for b in test_blocks:
    for t in session_templates[b]:
        if t not in vocab:
            unseen_test.add(t)
print(f"  Unseen in test: {len(unseen_test)} templates â†’ flagged anomalous directly")


def encode_session(blk):
    """
    Encode session as int IDs.
    Unseen template â†’ 0 (PAD sentinel, flagged anomalous before model).
    """
    return [vocab.get(t, 0) for t in session_templates[blk]]

# =============================================================================
# CELL 5 â€” Sliding Windows
# =============================================================================
print("\n  Building sliding windows ...")


def build_windows(blocks, skip_anomalous=False):
    """Slide WINDOW_SIZE over each session. Returns (X, y, session_ids)."""
    X_list, y_list, sid_list = [], [], []
    for sid, blk in enumerate(blocks):
        if skip_anomalous and session_labels.get(blk, 0) == 1:
            continue
        seq = encode_session(blk)
        if len(seq) < WINDOW_SIZE + 1:
            continue
        seq_arr = np.array(seq, dtype=np.int64)
        wins = np.lib.stride_tricks.sliding_window_view(seq_arr, WINDOW_SIZE)[:-1]
        targets = seq_arr[WINDOW_SIZE:]
        X_list.append(wins)
        y_list.append(targets)
        sid_list.append(np.full(len(targets), sid, dtype=np.int64))
    if not X_list:
        return (np.empty((0, WINDOW_SIZE), np.int64),
                np.empty((0,), np.int64),
                np.empty((0,), np.int64))
    return (np.concatenate(X_list, axis=0),
            np.concatenate(y_list, axis=0),
            np.concatenate(sid_list, axis=0))


# Train windows: normal sessions only [Du2017 Â§5.2]
X_tr, y_tr, _ = build_windows(train_normal, skip_anomalous=False)

# FIX 7: Val windows from NORMAL val sessions only (for training loss monitoring)
# k selection uses score_sessions_topk on ALL val sessions (Cell 9)
X_vl, y_vl, _ = build_windows(val_blocks, skip_anomalous=True)

print(f"  Train windows : {len(X_tr):,}")
print(f"  Val   windows : {len(X_vl):,}  (normal val sessions, for loss monitoring)")


class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).long()
        self.y = torch.from_numpy(y).long()

    def __len__(self): return len(self.X)

    def __getitem__(self, i): return self.X[i], self.y[i]


train_dl = DataLoader(
    WindowDataset(X_tr, y_tr),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    pin_memory=(DEVICE.type == 'cuda'),
)

# Val dataloader for loss monitoring
val_dl = DataLoader(
    WindowDataset(X_vl, y_vl),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    pin_memory=(DEVICE.type == 'cuda'),
)
del X_tr, y_tr, X_vl, y_vl; gc.collect()

# =============================================================================
# CELL 6 â€” Model (paper-exact architecture) [Du2017 Â§5.1]
# =============================================================================
print("\n  Building DeepLog model ...")


class DeepLog(nn.Module):
    """
    Unidirectional stacked LSTM for next-key prediction [Du2017].
    Embedding â†’ 2-layer LSTM â†’ Linear(hidden â†’ vocab_size)
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        emb       = self.embedding(x)      # (B, T, E)
        out, _    = self.lstm(emb)         # (B, T, H)
        last      = out[:, -1, :]          # (B, H)
        return self.fc(last)               # (B, V)


model = DeepLog(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parameters  : {n_params:,}  |  VOCAB_SIZE={VOCAB_SIZE}")

# =============================================================================
# CELL 7 â€” Scoring Function (needed during training for val F1)
#
# FIX 1 (full implementation):
#   Two-signal detection:
#     - Any session containing an unseen template (0) is immediately flagged as anomalous.
#     - Other sessions (with only known templates) are evaluated by checking if any 
#       window's next-key is in the top-k predictions of the LSTM.
# =============================================================================
use_amp = DEVICE.type == 'cuda'


def score_sessions_topk(blocks, k_values, batch_size=8192):
    """
    Returns dict: k â†’ np.array(len(blocks),) of 0/1 anomaly flags.

    Two-signal detection protocol:
      1. Any session containing an unseen template (PAD=0) â†’ immediately anomalous (100% anomaly).
      2. Any session with only known templates and len >= WINDOW_SIZE + 1 â†’ run LSTM top-k check.
      3. Any session with only known templates and len < WINDOW_SIZE + 1 â†’ normal.
    """
    model.eval()
    
    num_blocks = len(blocks)
    results = {k: np.zeros(num_blocks, dtype=np.int32) for k in k_values}
    
    unseen_indices = []
    lstm_indices = []
    lstm_blocks = []
    
    for sid, blk in enumerate(blocks):
        seq = encode_session(blk)
        if 0 in seq:
            unseen_indices.append(sid)
        else:
            if len(seq) >= WINDOW_SIZE + 1:
                lstm_indices.append(sid)
                lstm_blocks.append(seq)
            # Short and known-only is left as 0 (normal)
            
    for sid in unseen_indices:
        for k in k_values:
            results[k][sid] = 1
            
    if not lstm_blocks:
        return results
        
    all_win, all_true, all_sid = [], [], []
    for internal_idx, seq in enumerate(lstm_blocks):
        seq_arr = np.array(seq, dtype=np.int64)
        wins = np.lib.stride_tricks.sliding_window_view(seq_arr, WINDOW_SIZE)[:-1]
        targets = seq_arr[WINDOW_SIZE:]
        all_win.append(wins)
        all_true.append(targets)
        original_sid = lstm_indices[internal_idx]
        all_sid.append(np.full(len(targets), original_sid, dtype=np.int64))
        
    all_win = np.concatenate(all_win, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    all_sid = np.concatenate(all_sid, axis=0)
    
    k_top = max(k_values)
    
    ds = torch.utils.data.TensorDataset(torch.from_numpy(all_win).long())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    topk_list = []
    with torch.no_grad():
        for (xb,) in dl:
            if use_amp:
                with autocast(device_type='cuda'):
                    logits = model(xb.to(DEVICE))
            else:
                logits = model(xb.to(DEVICE))
            topk_list.append(logits.topk(k_top, dim=-1).indices.cpu().numpy())
    topk_all = np.concatenate(topk_list, axis=0)  # (N_windows, k_top)
    
    for k in k_values:
        top_k = topk_all[:, :k]
        anom_win = ~(top_k == all_true[:, None]).any(axis=1)
        
        sess_anom = np.zeros(num_blocks, dtype=np.int32)
        np.maximum.at(sess_anom, all_sid, anom_win.astype(np.int32))
        
        results[k] = np.maximum(results[k], sess_anom)
        
    return results

# =============================================================================
# CELL 8 â€” Training with Early Stopping & LR Scheduler
#
# FIX 8:  Epochs 30, early stopping on val F1 (patience=5)
# FIX 9:  ReduceLROnPlateau on val loss
# FIX 10: Class-weighted CrossEntropyLoss for template imbalance
# FIX 11: Best model checkpoint saved by val F1
# =============================================================================
print(f"\n  Training up to {EPOCHS} epochs (early stop patience={PATIENCE}) ...")

# FIX 10: Standard CrossEntropyLoss to preserve true probability transitions
# (Note: Class weighting on next-event prediction is counter-productive because
# it skews the model's transition probabilities, leading to massive false positives on normal sequences).
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# FIX 9: LR scheduler â€” reduce on plateau of val loss
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5
)

scaler = GradScaler(device='cuda') if use_amp else None

train_losses = []
val_losses = []
val_f1_history = []
best_val_f1 = 0.0
best_epoch = 0
no_improve_count = 0
t_train = time.time()

# FIX 7: Pre-compute val labels for F1 scoring on FULL val set
val_labels = np.array([session_labels.get(b, 0) for b in val_blocks])

for epoch in range(1, EPOCHS + 1):
    # â”€â”€ Train â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model.train()
    ep_loss = 0.0; n_total = 0
    for X_batch, y_batch in train_dl:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type='cuda'):
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            # Gradient clipping [Du2017] â€” stabilize LSTM training
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        ep_loss += loss.item() * len(X_batch)
        n_total += len(X_batch)

    avg_train_loss = ep_loss / n_total
    train_losses.append(avg_train_loss)

    # â”€â”€ Val loss (on normal val windows â€” for LR scheduler) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model.eval()
    val_loss_sum = 0.0; val_n = 0
    with torch.no_grad():
        for X_batch, y_batch in val_dl:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            if use_amp:
                with autocast(device_type='cuda'):
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
            else:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
            val_loss_sum += loss.item() * len(X_batch)
            val_n += len(X_batch)
    avg_val_loss = val_loss_sum / val_n if val_n > 0 else 0.0
    val_losses.append(avg_val_loss)

    # FIX 9: Step LR scheduler on val loss
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    # FIX 6+7+8: Compute val F1 on FULL val set (both normal+anomaly sessions)
    # Using K_MONITOR as a fixed mid-range k for training monitoring
    val_results_monitor = score_sessions_topk(val_blocks, [K_MONITOR], batch_size=8192)
    val_preds_monitor = val_results_monitor[K_MONITOR]
    epoch_val_f1 = f1_score(val_labels, val_preds_monitor, zero_division=0)
    val_f1_history.append(epoch_val_f1)

    # FIX 11: Save best model by val F1
    improved = ""
    if epoch_val_f1 > best_val_f1:
        best_val_f1 = epoch_val_f1
        best_epoch = epoch
        no_improve_count = 0
        torch.save(model.state_dict(), MODEL_PATH)
        improved = " â˜… saved"
    else:
        no_improve_count += 1

    print(f"  Epoch {epoch:2d}/{EPOCHS}  |  TrLoss: {avg_train_loss:.5f}  "
          f"VlLoss: {avg_val_loss:.5f}  ValF1(k={K_MONITOR}): {epoch_val_f1:.4f}  "
          f"LR: {current_lr:.6f}{improved}")

    # FIX 8: Early stopping on val F1
    if no_improve_count >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        break

print(f"\n  Training done ({time.time()-t_train:.0f}s)")
print(f"  Best epoch: {best_epoch} (Val F1={best_val_f1:.4f})")

# FIX 11: Reload best model checkpoint
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print(f"  Best model reloaded from {MODEL_PATH}")

# =============================================================================
# CELL 9 â€” k Selection on Val
#
# FIX 6: k selected by BEST F1 on FULL val set (normal + anomaly sessions)
# FIX 7: Val scoring includes ALL val sessions
# =============================================================================
print("\n  k Selection on validation set (F1 on full val) ...")
K_SEARCH = list(range(1, min(K_MAX, VOCAB_SIZE - 1) + 1))

print(f"  Scoring {len(val_blocks):,} val sessions (ALL â€” normal + anomaly) ...")
val_results = score_sessions_topk(val_blocks, K_SEARCH, batch_size=8192)

best_k = 1
best_k_f1 = 0.0
best_fpr = 1.0
val_norm_idx = [i for i, b in enumerate(val_blocks) if session_labels[b] == 0]
n_val_normal = len(val_norm_idx)
k_table = []

print(f"\n  {'k':>3}  {'Val F1':>8}  {'Val Prec':>10}  {'Val Rec':>10}  {'FPR':>10}  {'FP':>6}")
print(f"  {'-'*55}")
for k in K_SEARCH:
    preds_val = val_results[k]
    # FIX 6: F1 on full val set (both classes)
    k_f1   = f1_score(val_labels, preds_val, zero_division=0)
    k_prec = precision_score(val_labels, preds_val, zero_division=0)
    k_rec  = recall_score(val_labels, preds_val, zero_division=0)
    fps    = int(preds_val[val_norm_idx].sum())
    fpr    = fps / n_val_normal if n_val_normal > 0 else 0.0
    k_table.append({'k': k, 'F1': round(k_f1, 6), 'Precision': round(k_prec, 6),
                    'Recall': round(k_rec, 6), 'FP': fps, 'FPR': round(fpr, 6)})
    marker = " â† best" if k_f1 > best_k_f1 else ""
    print(f"  k={k:>2}  F1={k_f1:.6f}  P={k_prec:.4f}  R={k_rec:.4f}  "
          f"FPR={fpr:.6f}  FP={fps:,}{marker}")
    if k_f1 > best_k_f1:
        best_k_f1 = k_f1
        best_k = k
        best_fpr = fpr

print(f"\n  Best k = {best_k}  (Val F1 = {best_k_f1:.6f}, Val FPR = {best_fpr:.6f})")
pd.DataFrame(k_table).to_csv(f'{REPORT}/deeplog13_k_selection.csv', index=False)

# =============================================================================
# CELL 10 â€” Test Evaluation (touched EXACTLY ONCE)
# =============================================================================
print(f"\n  Test Evaluation (k={best_k}) ...")
test_labels_arr = np.array([session_labels.get(b, 0) for b in test_blocks], dtype=np.int32)

print(f"  Scoring {len(test_blocks):,} test sessions ...")
t_test = time.time()
test_results = score_sessions_topk(test_blocks, [best_k], batch_size=8192)
infer_time   = time.time() - t_test
print(f"  Done ({infer_time:.1f}s)")

preds     = test_results[best_k]
precision = precision_score(test_labels_arr, preds, zero_division=0)
recall    = recall_score(   test_labels_arr, preds, zero_division=0)
f1        = f1_score(       test_labels_arr, preds, zero_division=0)
mcc       = matthews_corrcoef(test_labels_arr, preds)
cm        = confusion_matrix(test_labels_arr, preds)
tn, fp, fn, tp = cm.ravel()
fpr_test = fp / (fp + tn) if (fp + tn) > 0 else 0.0
fnr_test = fn / (fn + tp) if (fn + tp) > 0 else 0.0

print(f"\n  {'='*60}")
print(f"  FINAL TEST RESULTS â€” DeepLog HDFS [Du2017]")
print(f"  {'='*60}")
print(f"  Test sessions : {len(test_blocks):,}  (norm={test_norm_n:,}, anom={test_anom_n:,})")
print(f"  Best k        : {best_k}  |  Vocab: {VOCAB_SIZE}")
print(f"  Best epoch    : {best_epoch}  (of {EPOCHS} max)")
print(f"  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
print(f"  TP = {tp:,}   FP = {fp:,}")
print(f"  FN = {fn:,}   TN = {tn:,}")
print(f"  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
print(f"  Precision     : {precision:.4f}")
print(f"  Recall        : {recall:.4f}")
print(f"  F1            : {f1:.4f}")
print(f"  MCC           : {mcc:.4f}")
print(f"  FPR (test)    : {fpr_test:.6f}")
print(f"  FNR (test)    : {fnr_test:.6f}")
print(f"  Paper F1      : 0.9600  |  Delta: {f1-0.96:+.4f}")
print(f"  {'='*60}")

if f1 >= 0.96:   grade = "âœ…  EXCELLENT (â‰¥ paper F1)"
elif f1 >= 0.95: grade = "âœ…  EXCELLENT"
elif f1 >= 0.90: grade = "ðŸŸ¡  GOOD"
elif f1 >= 0.80: grade = "ðŸŸ   ACCEPTABLE"
else:            grade = "ðŸ”´  NEEDS REVIEW"
print(f"  Grade: {grade}")

# =============================================================================
# CELL 11 â€” Save Results
# =============================================================================
actual_epochs = len(train_losses)
metrics_dict = dict(
    Dataset='HDFS', Model='DeepLog', Paper='Du2017',
    k=best_k, Vocab_size=VOCAB_SIZE, Window_size=WINDOW_SIZE,
    Hidden_size=HIDDEN_SIZE, Embed_dim=EMBED_DIM,
    Num_layers=NUM_LAYERS, Epochs_trained=actual_epochs,
    Best_epoch=best_epoch, Batch_size=BATCH_SIZE, LR=LR,
    Train_normal=len(train_normal), Val=len(val_blocks), Test=len(test_blocks),
    TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
    Precision=round(precision, 4), Recall=round(recall, 4),
    F1=round(f1, 4), MCC=round(mcc, 4),
    FPR_test=round(fpr_test, 6), FNR_test=round(fnr_test, 6),
    Val_F1_best_k=round(best_k_f1, 6),
    Val_FPR_best_k=round(best_fpr, 6),
    Infer_time_s=round(infer_time, 2),
    Paper_F1=0.96, Delta_F1=round(f1 - 0.96, 4),
)
pd.DataFrame([metrics_dict]).to_csv(f'{REPORT}/deeplog13_results.csv', index=False)
print(f"\n  Results â†’ {REPORT}/deeplog13_results.csv")

# =============================================================================
# CELL 12 â€” Plots
# =============================================================================
# 1. Training & Validation loss curves
fig, ax = plt.subplots(figsize=(8, 4))
eps = range(1, actual_epochs + 1)
ax.plot(eps, train_losses, 'o-', color='royalblue', lw=1.8, ms=4, label='Train Loss')
ax.plot(eps, val_losses, 's-', color='darkorange', lw=1.8, ms=4, label='Val Loss')
ax.axvline(best_epoch, color='green', linestyle='--', lw=1.5, alpha=0.7, label=f'Best epoch={best_epoch}')
ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-Entropy Loss')
ax.set_title('DeepLog Training & Validation Loss â€” HDFS [Du2017]', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f'{REPORT}/deeplog13_training_loss.png', dpi=150); plt.close()

# 2. Val F1 curve during training
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(eps, val_f1_history, 'D-', color='green', lw=1.8, ms=4, label=f'Val F1 (k={K_MONITOR})')
ax.axvline(best_epoch, color='crimson', linestyle='--', lw=1.5, alpha=0.7, label=f'Best epoch={best_epoch}')
ax.axhline(0.96, color='gold', linestyle='--', lw=1.5, label='Paper F1=0.96')
ax.set_xlabel('Epoch'); ax.set_ylabel('Val F1')
ax.set_title('DeepLog Val F1 During Training â€” HDFS [Du2017]', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f'{REPORT}/deeplog13_val_f1_curve.png', dpi=150); plt.close()

# 3. Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
ax.set_title(f'Confusion Matrix  k={best_k}  [Du2017]', fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(f'{REPORT}/deeplog13_confusion_matrix.png', dpi=150); plt.close()

# 4. k vs F1 (FIX 6: now showing F1, not just FPR)
k_df = pd.DataFrame(k_table)
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.bar(k_df['k'], k_df['F1'], color='steelblue', edgecolor='white', width=0.6, label='Val F1')
ax1.axvline(best_k, color='crimson', linestyle='--', lw=2, label=f'Best k={best_k}')
ax1.axhline(0.96, color='gold', linestyle='--', lw=1.5, label='Paper F1=0.96')
ax1.set_xlabel('k  (top-k predictions)'); ax1.set_ylabel('Validation F1')
ax1.set_title('k Selection â€” Validation F1  [Du2017 Protocol]', fontweight='bold')
ax1.legend(); ax1.grid(axis='y', alpha=0.3); plt.tight_layout()
plt.savefig(f'{REPORT}/deeplog13_k_selection.png', dpi=150); plt.close()

# 5. Metrics bar
fig, ax = plt.subplots(figsize=(6, 4))
metrics_bar = {'Precision': precision, 'Recall': recall, 'F1': f1, 'MCC': (mcc+1)/2}
ax.bar(list(metrics_bar.keys()), list(metrics_bar.values()),
       color=['#4C72B0','#55A868','#C44E52','#8172B2'], edgecolor='white', width=0.5)
ax.axhline(0.960, color='gold', lw=2, linestyle='--', label='Paper F1=0.96')
ax.set_ylim(0, 1.05); ax.set_ylabel('Score')
ax.set_title(f'DeepLog HDFS Metrics  (k={best_k})  [Du2017]', fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
for i, (name, val) in enumerate(metrics_bar.items()):
    ax.text(i, val + 0.01, f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{REPORT}/deeplog13_metrics_bar.png', dpi=150); plt.close()

print("  Plots saved.")

# =============================================================================
# FINAL RESULTS SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("  âœ…  FINAL RESULTS â€” DeepLog HDFS [Du2017]")
print("=" * 70)
print(f"\n  Vocab size   : {VOCAB_SIZE}  (train templates + PAD)")
print(f"  Window size  : {WINDOW_SIZE}  [Du2017 Â§5.1]")
print(f"  Train normal : {len(train_normal):,} sessions")
print(f"  Val          : {len(val_blocks):,}  sessions (full val used for k selection)")
print(f"  Test normal  : {test_norm_n:,}")
print(f"  Test anomaly : {test_anom_n:,}")
print(f"  Best k       : {best_k}")
print(f"  Best epoch   : {best_epoch} / {actual_epochs} trained")
print(f"\n  TP = {tp:,}   FP = {fp:,}")
print(f"  FN = {fn:,}   TN = {tn:,}")
print(f"\n  Precision    : {precision:.4f}")
print(f"  Recall       : {recall:.4f}")
print(f"  F1-Score     : {f1:.4f}")
print(f"  MCC          : {mcc:.4f}")
print(f"  FPR          : {fpr_test:.6f}")
print(f"  FNR          : {fnr_test:.6f}")
print(f"\n  Paper [Du2017] F1 = 0.960")
print(f"  Our F1           = {f1:.4f}  ({f1-0.96:+.4f})")
print(f"\n  Grade: {grade}")
print("\n  Fixes applied vs original NB13:")
print("    FIX 1:  Two-signal detection (session with unseen template -> 100% anomaly; others checked via LSTM top-k)")
print("    FIX 2:  Ratio split 80/10/10 (was hardcoded 4855/500)")
print("    FIX 3:  Stratified split preserves anomaly ratio")
print("    FIX 4:  torch.amp (not deprecated torch.cuda.amp)")
print("    FIX 5:  groupby() loading (~50Ã— faster than iterrows)")
print("    FIX 6:  k selected by F1 on FULL val (was FPR on normals only)")
print("    FIX 7:  Val scoring includes ALL sessions (normal + anomaly)")
print("    FIX 8:  Epochs 30 + early stopping (patience=5) on val F1")
print("    FIX 9:  ReduceLROnPlateau scheduler on val loss")
print("    FIX 10: Standard CrossEntropyLoss to preserve true probability transitions (class weighting disabled for maximum F1)")
print("    FIX 11: Best model saved by val F1, not last epoch")
print("=" * 70)


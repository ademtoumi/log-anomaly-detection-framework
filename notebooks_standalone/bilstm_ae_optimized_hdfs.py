# =============================================================================
# STANDALONE NOTEBOOK 12c â€” Optimized BiLSTM Autoencoder â€” HDFS Log Anomaly
#                           (Fully Independent, Zero Dependencies)
#
# âœ… ZERO dependencies â€” reads HDFS_Drain.csv directly.
# âœ… Builds HDFS sessions inline (BlockId grouping, stratified split).
# âœ… One dataset only (HDFS) â€” RAM stays safe on Kaggle T4/P100.
# âœ… Trains on NORMAL sessions only â€” reconstruction error for anomaly scoring.
# âœ… AttentionPool: hybrid context = (BiLSTM hidden state + attention over enc_out)
#
# Architecture (vs NB12 / NB12b):
#   Embedding â†’ BiLSTM Encoder â†’ AttentionPool + combine â†’ LSTM Decoder â†’ MSE
#   Key upgrade: dual context vector (hidden state + soft attention) combines
#   final hidden state with attention-weighted encoder output for richer
#   bottleneck representation.
#
# Split strategy (vs NB12 temporal 60/20/20):
#   Stratified 90/10 train/test, then 90/10 train/val (sklearn stratified)
#   Ensures every split preserves the ~2.9 % anomaly ratio.
#
# Scoring (vs NB12 pure MSE mean):
#   score = 0.5 * masked_mean_error + 0.5 * masked_max_error
#   Combined mean+max is more robust to partial anomalies within a session.
#
# Scheduler: ReduceLROnPlateau(mode='max', factor=0.7, patience=8)
#   Reduces LR when val F1 stagnates â€” adaptive LR for better convergence.
#
# References:
#   [Bekkouche2025_BiLSTM] â€” BiLSTM-AE, F1=0.993 on HDFS; paper baseline.
#   [Du2017_DeepLog]        â€” HDFS sessions naturally grouped by BlockId.
#   [Zhang2019_LogRobust]   â€” Attention over hidden states improves log detection.
#   [Guo2021_LogBERT]       â€” Bidirectional context for offline log analysis.
#
# Kaggle setup:
#   - Dataset : pfe-log-anomaly  (must contain HDFS_Drain.csv)
#   - Accelerator: GPU T4 or P100
#   - Estimated time: ~25 minutes (no Optuna; fixed best params from result-6)
#
# Results (result-6, Kaggle run):
#   F1=0.9571 | P=0.9177 | R=1.0000 | MCC=0.9567 | AUC=0.9990
#   BestValF1=0.9589 | Threshold=0.469545 | Epochs=31 | hidden=256, embed=64
#
# Paper target [Bekkouche2025_BiLSTM]: F1=0.9930  â†’ Delta = -0.0359
#   (Unsupervised AE gap vs supervised BiLSTM â€” expected & scientifically valid)
# =============================================================================

import os, gc, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    matthews_corrcoef, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# â”€â”€ Reproducibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# â”€â”€ Output dirs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
KAGGLE   = os.path.exists('/kaggle')
BASE_OUT = '/kaggle/working' if KAGGLE else 'result/results_bilstm_ae_optimized_hdfs'
REPORT   = f'{BASE_OUT}/pfe_report'
os.makedirs(REPORT, exist_ok=True)
print(f"{'Kaggle' if KAGGLE else 'Local'} environment | Outputs â†’ {REPORT}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 1 â€” Locate HDFS_Drain.csv
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
csv_candidates = [
    '/kaggle/input/pfe-log-anomaly/HDFS_Drain.csv',
    '/kaggle/input/datasets/toumiadem/pfe-log-anomaly/HDFS_Drain.csv',
    '/content/HDFS_Drain.csv',
    '/content/drive/MyDrive/pfe_log_anomaly_detection/data/raw/HDFS_Drain.csv',
    r'c:\Users\toumi\Desktop\(Anomaly detection)\data\raw\HDFS_Drain.csv',
    'HDFS_Drain.csv',
]
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

if csv_path is None:
    for sroot in ['/kaggle/input', '/content/drive/MyDrive', r'c:\Users\toumi']:
        if os.path.exists(sroot):
            for root, _, files in os.walk(sroot):
                if 'HDFS_Drain.csv' in files:
                    csv_path = os.path.join(root, 'HDFS_Drain.csv')
                    break
        if csv_path:
            break

if not csv_path:
    raise FileNotFoundError("HDFS_Drain.csv not found â€” check Kaggle dataset attachment")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 2 â€” Load & Group Sessions (Chunked, RAM-safe)
# Based on [Du2017_DeepLog]: HDFS sessions grouped naturally by BlockId.
# Chunked reading keeps peak RAM below Kaggle 16 GB limit.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\nLoading data from {csv_path}...")
block_events = {}   # BlockId â†’ list[template_str]
block_labels = {}   # BlockId â†’ int (max anomaly flag)
block_order  = []   # Temporal insertion order

chunk_num = 0
for chunk in pd.read_csv(csv_path, chunksize=500_000,
                          on_bad_lines='skip', low_memory=False):
    chunk_num += 1

    # Extract BlockId â€” prefer 'BlockId' column, else regex from 'log'
    if 'BlockId' in chunk.columns:
        chunk['_bid'] = chunk['BlockId'].astype(str).str.strip()
    else:
        chunk['_bid'] = chunk['log'].str.extract(r'(blk_-?\d+)')

    chunk = chunk.dropna(subset=['_bid'])

    # Anomaly label: any anomalous line â†’ session is anomalous
    # [Du2017_DeepLog]: session = anomalous if any line is anomalous
    lbl_col = 'Label' if 'Label' in chunk.columns else 'label'
    chunk['_anom'] = (chunk[lbl_col].astype(str).str.strip() != 'Normal').astype(int)

    chunk['template'] = chunk['template'].fillna('unknown').astype(str)

    # Efficient groupby instead of iterrows (faster for large chunks)
    for bid, grp in chunk.groupby('_bid'):
        if bid not in block_events:
            block_events[bid] = []
            block_labels[bid] = 0
            block_order.append(bid)
        block_events[bid].extend(grp['template'].tolist())
        block_labels[bid] = max(block_labels[bid], int(grp['_anom'].max()))

    if chunk_num % 5 == 0:
        print(f"    Chunk {chunk_num}: {len(block_events):,} blocks")
    del chunk
    gc.collect()

n_blocks = len(block_order)
n_anom   = sum(block_labels.values())
print(f"Total blocks : {n_blocks:,}")
print(f"Anomaly blocks: {n_anom:,} ({n_anom/n_blocks*100:.2f}%)")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 3 â€” Stratified Split 90 / 10 / 10
# [Bekkouche2025_BiLSTM]: stratified split preserves anomaly ratio.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\nApplying Stratified Split â€” 90/10 train/test, then 90/10 train/val")
all_labels_arr = np.array([block_labels[b] for b in block_order])

train_idx, test_idx = train_test_split(
    np.arange(n_blocks), test_size=0.10,
    random_state=SEED, stratify=all_labels_arr)
train_bids = [block_order[i] for i in train_idx]
test_bids  = [block_order[i] for i in test_idx]

train_labels_arr = np.array([block_labels[b] for b in train_bids])
train_idx2, val_idx2 = train_test_split(
    np.arange(len(train_bids)), test_size=0.10,
    random_state=SEED, stratify=train_labels_arr)
val_bids   = [train_bids[i] for i in val_idx2]
train_bids = [train_bids[i] for i in train_idx2]

print(f"Train : {len(train_bids):,} | Val: {len(val_bids):,} | Test: {len(test_bids):,}")
print(f"Train anomaly : {np.mean([block_labels[b] for b in train_bids]):.3%}")
print(f"Val   anomaly : {np.mean([block_labels[b] for b in val_bids]):.3%}")
print(f"Test  anomaly : {np.mean([block_labels[b] for b in test_bids]):.3%}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 4 â€” Vocabulary (train-only â€” no leakage)
# [Bekkouche2025_BiLSTM]: vocab built from TRAIN sessions only.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
all_templates = set()
for bid in train_bids:
    all_templates.update(block_events[bid])

vocab = {'<PAD>': 0, '<UNK>': 1}
for idx, t in enumerate(sorted(all_templates)):
    vocab[t] = idx + 2

MAX_SEQ_LEN = 75   # [Du2017]: most HDFS sessions < 50 events; 75 adds margin

def _encode(bids):
    """Encode a list of BlockIds into padded int32 sequences + labels."""
    seqs   = np.zeros((len(bids), MAX_SEQ_LEN), dtype=np.int32)
    labels = np.zeros(len(bids), dtype=np.int32)
    for i, bid in enumerate(bids):
        enc = [vocab.get(e, 1) for e in block_events[bid]]  # <UNK>=1 fallback
        sl  = min(len(enc), MAX_SEQ_LEN)
        seqs[i, :sl] = enc[:sl]
        labels[i]    = block_labels[bid]
    return seqs, labels

X_train, y_train = _encode(train_bids)
X_val,   y_val   = _encode(val_bids)
X_test,  y_test  = _encode(test_bids)
VOCAB_SIZE = len(vocab)
print(f"\nVocab size: {VOCAB_SIZE} (train-only, no test leakage)")

del block_events, block_labels
gc.collect()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 5 â€” Model Hyperparameters
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Best params from result-6 Kaggle run (31 epochs, ReduceLROnPlateau).
# No Optuna needed here â€” fixed best config for reproducibility.
PARAMS = {
    'embed_dim'  : 64,
    'hidden_size': 256,
    'num_layers' : 2,
    'dropout'    : 0.2,
    'lr'         : 0.001,
    'batch_size' : 256,
}
print(f"Model params : {PARAMS}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 6 â€” BiLSTM Autoencoder with AttentionPool
# Key difference vs NB12: dual context = hidden state + soft attention.
# [Zhang2019_LogRobust]: attention over ALL encoder outputs > last hidden only.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class AttentionPool(nn.Module):
    """Soft attention over encoder output â€” masks padding before softmax."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, enc_out, mask):
        # enc_out: (B, T, H*2)  mask: (B, T) 1=real 0=pad
        scores  = self.attn(enc_out).squeeze(-1)          # (B, T)
        scores  = scores.masked_fill(mask == 0, -1e4)     # ignore padding
        weights = torch.softmax(scores, dim=-1)            # (B, T)
        return (weights.unsqueeze(-1) * enc_out).sum(1)   # (B, H*2)


class BiLSTMAutoencoder(nn.Module):
    """
    Optimized BiLSTM Autoencoder for HDFS log anomaly detection.

    Encoder: Bidirectional LSTM â†’ dual context vector
      ctx_h  = relu(combine(h_fwd || h_bwd))          # final hidden state
      ctx_a  = relu(attn_proj(AttentionPool(enc_out))) # attention context
      ctx    = (ctx_h + ctx_a) / 2                    # fused bottleneck

    Decoder: Unidirectional LSTM conditioned on ctx.
    Loss: MSE between original and reconstructed embeddings.
    Score: 0.5 * masked_mean_error + 0.5 * masked_max_error

    References:
        [Bekkouche2025_BiLSTM] â€” BiLSTM-AE baseline, F1=0.993 on HDFS.
        [Zhang2019_LogRobust]  â€” Attention over ALL hidden states.
        [Guo2021_LogBERT]      â€” Bidirectional context for offline log analysis.
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Bidirectional encoder [Guo2021_LogBERT]
        self.encoder = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Hidden-state pathway: combine fwd + bwd directions
        self.combine   = nn.Linear(hidden_size * 2, hidden_size)

        # Attention pathway: soft attention over full encoder output
        self.attn_pool = AttentionPool(hidden_size * 2)
        self.attn_proj = nn.Linear(hidden_size * 2, hidden_size)

        # Decoder: unidirectional (future is unknown at decode time)
        self.decoder = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)

        # Project back to embedding space
        self.output_proj = nn.Linear(hidden_size, embed_dim)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x):
        B, T = x.size()
        mask  = (x != 0).float()                              # (B, T)
        emb   = self.dropout(self.embedding(x))               # (B, T, E)

        enc_out, (h_n, _) = self.encoder(emb)                 # h_n: (layers*2, B, H)

        # Hidden-state context (last BiLSTM layer)
        h_fwd = h_n[-2]                                        # (B, H)
        h_bwd = h_n[-1]                                        # (B, H)
        ctx_h = torch.relu(self.combine(torch.cat([h_fwd, h_bwd], dim=-1)))  # (B, H)

        # Attention context [Zhang2019_LogRobust]
        ctx_a = torch.relu(self.attn_proj(self.attn_pool(enc_out, mask)))    # (B, H)

        # Fused bottleneck
        ctx = (ctx_h + ctx_a) / 2.0                           # (B, H)

        # Decoder input: expand ctx across T timesteps
        dec_in = ctx.unsqueeze(1).expand(B, T, self.hidden_size)  # (B, T, H)
        h0 = ctx.unsqueeze(0)                                  # (1, B, H)
        c0 = torch.zeros_like(h0)

        decoded, _ = self.decoder(dec_in, (h0, c0))           # (B, T, H)
        recon = self.output_proj(decoded)                       # (B, T, E)

        return emb, recon, mask

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 7 â€” Scoring & Threshold Search
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compute_scores(mdl, X, batch_size=256):
    """Per-session anomaly score = 0.5*masked_mean + 0.5*masked_max MSE."""
    mdl.eval()
    scores = []
    dl = DataLoader(TensorDataset(torch.from_numpy(X).long()),
                    batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            emb, recon, mask = mdl(xb)

            err   = ((emb - recon) ** 2).mean(dim=-1)          # (B, T) per-token
            err_m = err * mask                                   # zero out padding

            valid   = mask.sum(dim=1).clamp(min=1)
            mean_sc = err_m.sum(dim=1) / valid                  # masked mean

            err_max = err.masked_fill(mask == 0, -1e4)
            max_sc  = err_max.max(dim=1).values.clamp(min=0)   # masked max

            sc = 0.5 * mean_sc + 0.5 * max_sc
            scores.append(sc.cpu().numpy().astype(np.float32))

    return np.concatenate(scores)


def find_best_threshold(val_sc, y_v, n=10000):
    """
    Grid search F1-optimal threshold on VALIDATION scores.

    CRITICAL: threshold searched on validation data only.
    Test set is never used in threshold selection.
    [Bekkouche2025_BiLSTM]: F1-sensitive threshold selection.

    Returns: (best_threshold, best_val_f1)
    """
    lo = float(np.percentile(val_sc, 0.1))
    hi = float(np.percentile(val_sc, 99.9))
    best_f1, best_thr = 0.0, lo
    for thr in np.linspace(lo, hi, n):
        preds = (val_sc > thr).astype(int)
        f1 = f1_score(y_v, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr), float(best_f1)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 8 â€” Training (Normal-only, ReduceLROnPlateau)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Train exclusively on normal sessions â€” reconstruction error separates anomalies.
# [Bekkouche2025_BiLSTM]: unsupervised reconstruction-based anomaly detection.
X_train_normal = X_train[y_train == 0]
print(f"\nTraining on {len(X_train_normal):,} normal blocks "
      f"({len(X_train_normal)/len(X_train)*100:.1f}% of train)")

mdl = BiLSTMAutoencoder(
    VOCAB_SIZE,
    PARAMS['embed_dim'],
    PARAMS['hidden_size'],
    PARAMS['num_layers'],
    PARAMS['dropout'],
).to(DEVICE)

crit  = nn.MSELoss()
opt   = torch.optim.AdamW(mdl.parameters(),
                           lr=PARAMS['lr'], weight_decay=1e-4)
# ReduceLROnPlateau on val F1 â€” reduces LR when F1 stagnates
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='max', factor=0.7, patience=8, min_lr=1e-5)
scaler = GradScaler(device='cuda' if DEVICE == 'cuda' else 'cpu')

dl = DataLoader(
    TensorDataset(torch.from_numpy(X_train_normal).long()),
    batch_size=PARAMS['batch_size'], shuffle=True, num_workers=0)

best_f1    = 0.0
best_thr   = 0.0
best_state = None
no_imp     = 0
MAX_EPOCHS = 50
PATIENCE   = 10
history    = {'loss': [], 'val_f1': []}

print("\nTraining BiLSTM-AE (Optimized â€” AttentionPool + ReduceLROnPlateau)...")
for epoch in range(1, MAX_EPOCHS + 1):
    mdl.train()
    epoch_loss = 0.0

    for (xb,) in dl:
        xb = xb.to(DEVICE)
        opt.zero_grad()
        with autocast(device_type='cuda' if DEVICE == 'cuda' else 'cpu'):
            emb, recon, _ = mdl(xb)
            loss = crit(recon, emb.detach())
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        epoch_loss += loss.item()

    # Validation: compute scores â†’ F1-optimal threshold
    # CRITICAL: threshold searched on val only, never on test
    v_sc     = compute_scores(mdl, X_val, PARAMS['batch_size'])
    thr, vf1 = find_best_threshold(v_sc, y_val)
    avg_loss  = epoch_loss / len(dl)

    sched.step(vf1)   # ReduceLROnPlateau steps on val F1
    history['loss'].append(avg_loss)
    history['val_f1'].append(vf1)

    if vf1 > best_f1:
        best_f1    = vf1
        best_thr   = thr
        best_state = {k: v.clone() for k, v in mdl.state_dict().items()}
        no_imp     = 0
    else:
        no_imp += 1

    print(f"Epoch {epoch:>3}/{MAX_EPOCHS} | Loss={avg_loss:.5f} | "
          f"ValF1={vf1:.4f} | Best={best_f1:.4f} | "
          f"LR={opt.param_groups[0]['lr']:.2e}")

    if no_imp >= PATIENCE:
        print(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
        break

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 9 â€” Final Test Evaluation (test set touched exactly once)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
mdl.load_state_dict(best_state)
t_sc  = compute_scores(mdl, X_test, PARAMS['batch_size'])
preds = (t_sc > best_thr).astype(int)

p   = precision_score(y_test, preds, pos_label=1, zero_division=0)
r   = recall_score(y_test, preds, pos_label=1, zero_division=0)
f1  = f1_score(y_test, preds, pos_label=1, zero_division=0)
mcc = matthews_corrcoef(y_test, preds)
try:
    auc = roc_auc_score(y_test, t_sc)
except Exception:
    auc = 0.0

PAPER_F1 = 0.9930   # [Bekkouche2025_BiLSTM]: supervised BiLSTM on HDFS
delta    = f1 - PAPER_F1

print("\n" + "=" * 57)
print("  FINAL TEST RESULTS â€” BiLSTM-AE HDFS (Optimized)")
print("=" * 57)
print(classification_report(
    y_test, preds, target_names=['Normal', 'Anomaly'], digits=4))
print(f"Precision : {p:.4f}")
print(f"Recall    : {r:.4f}")
print(f"F1        : {f1:.4f}")
print(f"MCC       : {mcc:.4f}")
print(f"AUC-ROC   : {auc:.4f}")
print(f"Threshold : {best_thr:.6f}")
print(f"Best ValF1: {best_f1:.4f}")
print(f"Paper F1  : {PAPER_F1:.4f}  [Bekkouche2025_BiLSTM]")
print(f"Delta     : {delta:+.4f}  (unsupervised vs supervised â€” expected gap)")
print("=" * 57)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 10 â€” Save Results
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
results_df = pd.DataFrame([{
    'Model'        : 'BiLSTM-AE-Optimized',
    'Dataset'      : 'HDFS',
    'Split'        : 'Stratified-90/10',
    'Precision'    : round(p, 4),
    'Recall'       : round(r, 4),
    'F1'           : round(f1, 4),
    'MCC'          : round(mcc, 4),
    'AUC'          : round(auc, 4),
    'BestThreshold': round(best_thr, 6),
    'BestValF1'    : round(best_f1, 4),
    'Epochs'       : len(history['loss']),
    'PaperF1'      : PAPER_F1,
    'Delta'        : round(delta, 4),
    'hidden_size'  : PARAMS['hidden_size'],
    'embed_dim'    : PARAMS['embed_dim'],
    'num_layers'   : PARAMS['num_layers'],
    'dropout'      : PARAMS['dropout'],
    'lr'           : PARAMS['lr'],
    'batch_size'   : PARAMS['batch_size'],
    'Scoring'      : '0.5*mean+0.5*max',
    'Scheduler'    : 'ReduceLROnPlateau(max,0.7,p8)',
}])
csv_out = f'{REPORT}/bilstm_ae_optimized_hdfs_results.csv'
results_df.to_csv(csv_out, index=False)
print(f"Results saved â†’ {csv_out}")

model_out = f'{REPORT}/bilstm_ae_optimized_hdfs.pt'
torch.save(best_state, model_out)
print(f"Model saved  â†’ {model_out}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 11 â€” Plots
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Confusion Matrix
cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'],
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_title('BiLSTM-AE (Optimized) Confusion Matrix â€” HDFS', fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
cm_out = f'{REPORT}/bilstm_ae_optimized_confusion_matrix.png'
plt.savefig(cm_out, dpi=300)
plt.show(); plt.close()
print(f"Saved â†’ {cm_out}")

# 2. ROC Curve
fpr_arr, tpr_arr, _ = roc_curve(y_test, t_sc)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_arr, tpr_arr, 'b-', lw=2, label=f'AUC={auc:.4f}')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve â€” BiLSTM-AE (Optimized) HDFS', fontweight='bold')
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout()
roc_out = f'{REPORT}/bilstm_ae_optimized_roc_curve.png'
plt.savefig(roc_out, dpi=300)
plt.show(); plt.close()
print(f"Saved â†’ {roc_out}")

# 3. Training Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['loss'], color='royalblue', lw=1.5, label='Train Loss')
ax1.set_title('Training Loss â€” BiLSTM-AE Optimized HDFS')
ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(history['val_f1'], color='orange', lw=1.5, label='Val F1')
ax2.axhline(best_f1, linestyle='--', color='red', lw=1.2,
            label=f'Best={best_f1:.4f}')
ax2.set_title('Validation F1 (optimal threshold) â€” HDFS')
ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
curve_out = f'{REPORT}/bilstm_ae_optimized_training_curves.png'
plt.savefig(curve_out, dpi=300)
plt.show(); plt.close()
print(f"Saved â†’ {curve_out}")

# 4. Score Distribution (Normal vs Anomaly)
scores_normal = t_sc[y_test == 0]
scores_anom   = t_sc[y_test == 1]
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(scores_normal, bins=100, alpha=0.6, color='steelblue',
        label=f'Normal  (n={len(scores_normal):,})', density=True)
ax.hist(scores_anom, bins=100, alpha=0.6, color='crimson',
        label=f'Anomaly (n={len(scores_anom):,})',  density=True)
ax.axvline(best_thr, color='black', linestyle='--', lw=1.8,
           label=f'Threshold={best_thr:.4f}')
ax.set_title('Anomaly Score Distribution â€” BiLSTM-AE (Optimized) HDFS',
             fontweight='bold')
ax.set_xlabel('Score (0.5Â·mean + 0.5Â·max MSE)')
ax.set_ylabel('Density')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
dist_out = f'{REPORT}/bilstm_ae_optimized_score_dist.png'
plt.savefig(dist_out, dpi=300)
plt.show(); plt.close()
print(f"Saved â†’ {dist_out}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 12 â€” Summary
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 57)
print("  NOTEBOOK 12c â€” BiLSTM-AE OPTIMIZED HDFS â€” COMPLETE")
print("=" * 57)
print(f"  F1={f1:.4f} | P={p:.4f} | R={r:.4f} | MCC={mcc:.4f} | AUC={auc:.4f}")
print(f"  Paper F1 [Bekkouche2025_BiLSTM] : {PAPER_F1:.4f} (supervised)")
print(f"  Delta (unsupervised vs supervised): {delta:+.4f}")
print(f"\n  Key design choices vs NB12 / NB12b:")
print(f"    âœ… Dual context: hidden state + AttentionPool (fused bottleneck)")
print(f"    âœ… Score: 0.5Â·mean + 0.5Â·max masked MSE per session")
print(f"    âœ… Scheduler: ReduceLROnPlateau(mode='max') on val F1")
print(f"    âœ… Split: Stratified 90/10/10 preserves anomaly ratio")
print(f"    âœ… Threshold: F1-optimal on val only (10,000 grid points)")
print(f"\n  Paper citations:")
print(f"    [Bekkouche2025_BiLSTM] â€” BiLSTM-AE baseline, F1=0.993 HDFS")
print(f"    [Du2017_DeepLog]       â€” HDFS sessions grouped by BlockId")
print(f"    [Zhang2019_LogRobust]  â€” Attention over ALL hidden states")
print(f"    [Guo2021_LogBERT]      â€” Bidirectional context for offline logs")
print("=" * 57)


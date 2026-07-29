import os, gc, json, pathlib, time, random, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

KAGGLE = os.path.exists('/kaggle')
# NOTE: This script prints results to console but does not save files.
# The v2 improved result is confirmed in result/results_LSTMAE_HDFS_v2.
# To save results, add file-writing code using OUTPUT_DIR.
OUTPUT_DIR = '/kaggle/working' if KAGGLE else 'result/results_LSTMAE_HDFS_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Try to mount Colab drive only if we are not on Kaggle
if not KAGGLE:
    try:
        import google.colab
        from google.colab import drive
        print("Detected Google Colab. Mounting Drive...")
        try:
            drive.mount('/content/drive', force_remount=False)
        except Exception as mount_err:
            print(f"Drive mount failed: {mount_err}. Proceeding to look for dataset...")
    except ImportError:
        pass

# Find CSV
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

print(f"Loading data from {csv_path}...")
block_events, block_labels, block_order = {}, {}, []
chunk_num = 0
for chunk in pd.read_csv(csv_path, chunksize=500_000, on_bad_lines='skip', low_memory=False):
    chunk_num += 1
    if 'BlockId' in chunk.columns:
        chunk['_bid'] = chunk['BlockId'].astype(str).str.strip()
    else:
        chunk['_bid'] = chunk['log'].str.extract(r'(blk_-?\d+)')
    chunk = chunk.dropna(subset=['_bid'])
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
print(f"Total blocks: {n_blocks:,}")

# Fix 1 — Change the data split to match the paper exactly
print("Applying Uniform Split (random shuffle), train ratio=0.9, 10% of train as val")
random.shuffle(block_order)
i1 = int(n_blocks * 0.90)
train_bids = block_order[:i1]
test_bids  = block_order[i1:]

# Use 10% of train as validation
i_val = int(len(train_bids) * 0.90)
val_bids   = train_bids[i_val:]
train_bids = train_bids[:i_val]

print(f"Train blocks: {len(train_bids)}")
print(f"Val blocks  : {len(val_bids)}")
print(f"Test blocks : {len(test_bids)}")

all_templates = set()
for bid in train_bids:
    all_templates.update(block_events[bid])
vocab = {'<PAD>': 0, '<UNK>': 1}
for idx, t in enumerate(sorted(all_templates)):
    vocab[t] = idx + 2

MAX_SEQ_LEN = 75
def _encode(bids):
    seqs   = np.zeros((len(bids), MAX_SEQ_LEN), dtype=np.int32)
    labels = np.zeros(len(bids), dtype=np.int32)
    for i, bid in enumerate(bids):
        enc = [vocab.get(e, 1) for e in block_events[bid]]
        sl  = min(len(enc), MAX_SEQ_LEN)
        seqs[i, :sl] = enc[:sl]
        labels[i] = block_labels[bid]
    return seqs, labels

X_train, y_train = _encode(train_bids)
X_val,   y_val   = _encode(val_bids)
X_test,  y_test  = _encode(test_bids)
VOCAB_SIZE = len(vocab)
del block_events, block_labels; gc.collect()

# Fix 2 — Use the correct architecture params
NB12_PARAMS = {
    'embed_dim': 64,
    'hidden_size': 256,
    'num_layers': 2,
    'dropout': 0.2,
    'lr': 0.001,
    'batch_size': 256
}
print(f"Model params: {NB12_PARAMS}")

class BiLSTMAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        self.combine_directions = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, embed_dim)

    def forward(self, x):
        B, T = x.size(0), x.size(1)
        embedded = self.embedding(x)
        _, (h_n, _) = self.encoder(embedded)
        h_fwd = h_n[-2]; h_bwd = h_n[-1]
        ctx = torch.relu(self.combine_directions(torch.cat([h_fwd, h_bwd], dim=-1)))
        h0 = ctx.unsqueeze(0)
        c0 = torch.zeros_like(h0)
        dec_in  = torch.zeros(B, T, self.hidden_size, device=x.device)
        decoded, _ = self.decoder(dec_in, (h0, c0))
        recon = self.output_proj(decoded)
        return embedded, recon

def compute_raw_errors(mdl, X, batch_size=256):
    mdl.eval()
    all_pos, all_mask = [], []
    dl = DataLoader(TensorDataset(torch.from_numpy(X).long()), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            emb, recon = mdl(xb)
            per_pos = ((emb - recon) ** 2).mean(dim=-1)
            mask    = (xb != 0).float()
            all_pos.append((per_pos * mask).cpu().numpy().astype(np.float32))
            all_mask.append(mask.cpu().numpy().astype(np.float32))
    return np.concatenate(all_pos), np.concatenate(all_mask)

# Fix 3 — Use the paper's threshold strategy exactly
def thr_paper(val_sc, y_v, n=5000):
    lo = float(np.percentile(val_sc, 0.1))
    hi = float(np.percentile(val_sc, 99.9))
    best_f1, best_thr = 0.0, lo
    for thr in np.linspace(lo, hi, n):
        preds = (val_sc > thr).astype(int)
        f1 = f1_score(y_v, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr), float(best_f1)

X_train_normal = X_train[y_train == 0]

print("Training Autoencoder...")
mdl = BiLSTMAutoencoder(VOCAB_SIZE, NB12_PARAMS['embed_dim'], NB12_PARAMS['hidden_size'],
                        NB12_PARAMS['num_layers'], NB12_PARAMS['dropout']).to(DEVICE)
crit = nn.MSELoss()
opt = torch.optim.AdamW(mdl.parameters(), lr=NB12_PARAMS['lr'], weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='max', factor=0.5, patience=5, min_lr=1e-5
)
scaler = GradScaler()
dl = DataLoader(TensorDataset(torch.from_numpy(X_train_normal).long()),
                batch_size=NB12_PARAMS['batch_size'], shuffle=True, num_workers=0)

best_f1, best_thr, best_state, no_imp = 0.0, 0.0, None, 0
max_epochs = 150
patience = 20

for epoch in range(1, max_epochs + 1):
    mdl.train(); epoch_loss = 0.0
    for (xb,) in dl:
        xb = xb.to(DEVICE); opt.zero_grad()
        with autocast():
            emb, recon = mdl(xb)
            loss = crit(recon, emb.detach())
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        epoch_loss += loss.item()
    
    vp, vm = compute_raw_errors(mdl, X_val, batch_size=NB12_PARAMS['batch_size'])
    v_sc = (vp.sum(axis=1) / vm.sum(axis=1).clip(min=1)) # mean aggregation
    
    thr, vf1 = thr_paper(v_sc, y_val)
    
    sched.step(vf1)
    
    if vf1 > best_f1:
        best_f1, best_thr = vf1, thr
        best_state = {k: v.clone() for k, v in mdl.state_dict().items()}
        no_imp = 0
    else:
        no_imp += 1
        
    print(f"Epoch {epoch:>2}/{max_epochs} | Loss={epoch_loss/len(dl):.5f} | ValF1={vf1:.4f} | Best={best_f1:.4f}")
    if no_imp >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

mdl.load_state_dict(best_state)

vp, vm = compute_raw_errors(mdl, X_test, batch_size=NB12_PARAMS['batch_size'])
t_sc = (vp.sum(axis=1) / vm.sum(axis=1).clip(min=1))

preds = (t_sc > best_thr).astype(int)
print("\n" + "="*50)
print("FINAL TEST RESULTS (Paper implementation):")
print("="*50)
p = precision_score(y_test, preds, pos_label=1, zero_division=0)
r = recall_score(y_test, preds, pos_label=1, zero_division=0)
f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)

print(f"Precision: {p:.4f}")
print(f"Recall:    {r:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("="*50)

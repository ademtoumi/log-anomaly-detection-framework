
# ======================================================================
# # HierAttn-Block — Notebook 1: Data Pipeline
# **Steps:** Environment Setup → Session Construction → Feature Extraction → Dataset & DataLoaders
# 
# > Output: `features.pkl` saved to `/kaggle/working/hierattn_output/cache/`
# > Next: Run `nb2_baselines.ipynb`
# ======================================================================


# ======================================================================
# ## Step 1 — Environment Setup
# ======================================================================

import os, re, math, random, time, warnings, json, pickle
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split

# ── Seed ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print('='*60)
print('HierAttn-Block  —  HDFS Log Anomaly Detection')
print(f'PyTorch  : {torch.__version__}')
print(f'Device   : {DEVICE}')
print(f'Seed     : {SEED}')
print('='*60)


# ── Paths ─────────────────────────────────────────────────────────────────────
KAGGLE = os.path.exists('/kaggle/working')
if KAGGLE:
    input_dirs = [d for d in os.listdir('/kaggle/input') if os.path.isdir(f'/kaggle/input/{d}')]
    assert input_dirs, 'No dataset found in /kaggle/input'
    DATA_DIR   = f'/kaggle/input/{input_dirs[0]}'
    OUTPUT_DIR = '/kaggle/working/hierattn_output'
else:
    DATA_DIR   = './Dataset'
    OUTPUT_DIR = 'result/results_DeepLogEnhanced_HDFS_v2/hierattn_output'

CACHE_DIR  = os.path.join(OUTPUT_DIR, 'cache')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
MODEL_DIR  = os.path.join(OUTPUT_DIR, 'models')
for d in [OUTPUT_DIR, CACHE_DIR, FIGURE_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# Auto-detect CSV
csv_candidates = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.csv')]
assert csv_candidates, f'No CSV found in {DATA_DIR}'
CSV_PATH = os.path.join(DATA_DIR, csv_candidates[0])

print(f'DATA_DIR   : {DATA_DIR}')
print(f'CSV_PATH   : {CSV_PATH}')
print(f'OUTPUT_DIR : {OUTPUT_DIR}')

# ── Hyperparameters ───────────────────────────────────────────────────────────
MAX_LEN    = 32
BATCH_SIZE = 128
MAX_EPOCHS = 50
LR         = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE   = 7



# ======================================================================
# ## Step 2 — Load CSV & Session Construction
# ======================================================================

print(f'Loading {CSV_PATH} ...')
df_raw = pd.read_csv(CSV_PATH, nrows=None, on_bad_lines='skip', low_memory=False)
print(f'Raw shape: {df_raw.shape}')

# Normalise column names
df_raw.columns = [c.lower().strip() for c in df_raw.columns]
for src, tgt in [('Label','label'),('Template','template'),
                  ('Log','log'),('Content','log'),('content','log')]:
    if src in df_raw.columns and tgt not in df_raw.columns:
        df_raw.rename(columns={src: tgt}, inplace=True)

print(f'Columns  : {list(df_raw.columns)}')
print(f'\nLabel distribution:')
print(df_raw['label'].value_counts())


# ── Extract Block ID ──────────────────────────────────────────────────────────
df_raw['block_id'] = df_raw['log'].str.extract(r'(blk_-?\d+)')
df_raw = df_raw.dropna(subset=['block_id']).copy()
print(f'Rows with block_id: {len(df_raw):,}')

# ── Encode labels ─────────────────────────────────────────────────────────────
df_raw['is_anomaly'] = (df_raw['label'].str.strip() != 'Normal').astype(int)

# ── Template → integer ID ─────────────────────────────────────────────────────
unique_templates = df_raw['template'].dropna().unique().tolist()
template2id = {'<PAD>': 0, '<UNK>': 1}
for i, t in enumerate(sorted(unique_templates)):
    template2id[t] = i + 2
VOCAB_SIZE = len(template2id)
df_raw['event_id'] = df_raw['template'].map(template2id).fillna(1).astype(int)
print(f'Vocabulary size: {VOCAB_SIZE} templates')


# ── Parse numeric fields ──────────────────────────────────────────────────────
def parse_log_fields(log_str):
    size = 0
    m = re.search(r'(\d+)\s*(?:bytes?|B\b)', log_str, re.I)
    if m:
        size = int(m.group(1))
    else:
        nums = re.findall(r'\b(\d{6,})\b', log_str)
        if nums:
            size = int(nums[-1])
    ip = 0
    m_ip = re.search(r'(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})', log_str)
    if m_ip:
        octs = [int(x) for x in m_ip.groups()]
        ip = (octs[0]*256**3 + octs[1]*256**2 + octs[2]*256 + octs[3]) % 1000
    thread = 0
    m_t = re.search(r'(?:thread|tid)[:\s]*(\d+)', log_str, re.I)
    if m_t:
        thread = int(m_t.group(1))
    else:
        m_t2 = re.search(r'\b(\d{4,6})\b', log_str)
        if m_t2:
            thread = int(m_t2.group(1))
    return size, ip, thread

print('Parsing log fields (size, ip, thread) ...')
parsed = df_raw['log'].apply(parse_log_fields)
df_raw['size_val']  = parsed.apply(lambda x: x[0])
df_raw['ip_hash']   = parsed.apply(lambda x: x[1])
df_raw['thread_id'] = parsed.apply(lambda x: x[2])
df_raw['row_idx']   = np.arange(len(df_raw))
print('Parsing done.')


# ── Group into sessions ───────────────────────────────────────────────────────
print('Grouping rows into sessions by block_id ...')

def agg_session(grp):
    return pd.Series({
        'event_ids':   list(grp['event_id']),
        'templates':   list(grp['template'].fillna('<UNK>')),
        'size_vals':   list(grp['size_val']),
        'ip_hashes':   list(grp['ip_hash']),
        'thread_ids':  list(grp['thread_id']),
        'row_indices': list(grp['row_idx']),
        'label':       int(grp['is_anomaly'].max()),
    })

sessions = df_raw.groupby('block_id').apply(agg_session).reset_index()

n_total   = len(sessions)
n_normal  = (sessions['label'] == 0).sum()
n_anomaly = (sessions['label'] == 1).sum()
print(f'Total sessions : {n_total:,}')
print(f'Normal         : {n_normal:,}  ({100*n_normal/n_total:.1f}%)')
print(f'Anomaly        : {n_anomaly:,}  ({100*n_anomaly/n_total:.1f}%)')



# ======================================================================
# ## Step 3 — Feature Extraction
# ======================================================================

# ── Top-5 templates for structural features ───────────────────────────────────
template_freq = Counter(df_raw['template'].tolist())
top5_templates = [t for t, _ in template_freq.most_common(5)]
top5_ids       = [template2id.get(t, 1) for t in top5_templates]
print(f'Top-5 template IDs: {top5_ids}')

thread_max = df_raw['thread_id'].max()
if thread_max == 0:
    thread_max = 1

# ── Vectorised sinusoidal encoding ────────────────────────────────────────────
def sinusoidal_encoding(positions, d_model=32):
    positions = np.array(positions, dtype=np.float32)
    dims      = np.arange(0, d_model, 2, dtype=np.float32)
    denoms    = 10000 ** (dims / d_model)
    pe        = np.zeros((len(positions), d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(positions[:, None] / denoms[None, :])
    half = len(pe[0, 1::2])
    pe[:, 1::2] = np.cos(positions[:, None] / denoms[None, :half])
    return pe


# ── Find allocate template ID ─────────────────────────────────────────────────
allocate_id = None
for t, tid in template2id.items():
    if 'allocat' in t.lower():
        allocate_id = tid
        break
print(f'Allocate template ID: {allocate_id}')

def extract_features(row):
    evs   = row['event_ids']
    sizes = row['size_vals']
    ips   = row['ip_hashes']
    thrs  = row['thread_ids']
    ridxs = row['row_indices']
    n     = len(evs)
    seq_len = min(n, MAX_LEN)

    event_ids = np.zeros(MAX_LEN, dtype=np.int64)
    event_ids[:seq_len] = evs[:seq_len]

    param_feats = np.zeros((MAX_LEN, 3), dtype=np.float32)
    for i in range(seq_len):
        param_feats[i, 0] = math.log(sizes[i] + 1)
        param_feats[i, 1] = (ips[i] % 1000) / 1000.0
        param_feats[i, 2] = min(thrs[i], thread_max) / thread_max

    time_deltas = np.array([float(ridxs[i] - ridxs[0]) for i in range(seq_len)], dtype=np.float32)
    sin_padded  = np.zeros((MAX_LEN, 32), dtype=np.float32)
    sin_padded[:seq_len] = sinusoidal_encoding(time_deltas, d_model=32)

    attention_mask = np.zeros(MAX_LEN, dtype=np.float32)
    attention_mask[:seq_len] = 1.0

    ev_arr = np.array(evs)
    struct = np.zeros(11, dtype=np.float32)
    for k, tid in enumerate(top5_ids):
        struct[k] = float(np.sum(ev_arr == tid))
    struct[5] = float(np.std(sizes)) if len(sizes) > 1 else 0.0
    struct[6] = float(len(set(ips)))
    struct[7] = float(ridxs[-1] - ridxs[0]) if n > 1 else 0.0
    struct[8] = float(max(ridxs[i+1] - ridxs[i] for i in range(n-1))) if n > 1 else 0.0
    struct[9] = 0.0 if (allocate_id is not None and allocate_id in ev_arr) else 1.0

    repl_count = sum(1 for t in row['templates'] if 'replicat' in t.lower())
    struct[10] = 1.0 if repl_count != 3 else 0.0

    return {
        'event_ids':      event_ids,
        'param_feats':    param_feats,
        'sin_time':       sin_padded,
        'struct_feats':   struct,
        'attention_mask': attention_mask,
        'label':          int(row['label']),
        'repl_count':     float(repl_count),
        'missing_alloc':  int(struct[9]),
        'repl_neq3':      int(struct[10]),
    }

print('Extracting features for all sessions ...')
features = [extract_features(sessions.iloc[i]) for i in range(len(sessions))]
print(f'Done: {len(features)} feature dicts')



# ======================================================================
# ## Step 4 — PyTorch Dataset & DataLoaders
# ======================================================================

class HDFSDataset(Dataset):
    def __init__(self, feat_list):
        self.data = feat_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        return (
            torch.tensor(d['event_ids'],      dtype=torch.long),
            torch.tensor(d['param_feats'],    dtype=torch.float32),
            torch.tensor(d['sin_time'],       dtype=torch.float32),
            torch.tensor(d['struct_feats'],   dtype=torch.float32),
            torch.tensor(d['attention_mask'], dtype=torch.float32),
            torch.tensor(d['label'],          dtype=torch.long),
            torch.tensor(d['repl_count'],     dtype=torch.float32),
        )

# ── Stratified 70/10/20 split ─────────────────────────────────────────────────
labels_all = np.array([f['label'] for f in features])
indices    = np.arange(len(features))

idx_tv, idx_test = train_test_split(
    indices, test_size=0.20, stratify=labels_all, random_state=SEED)
labels_tv = labels_all[idx_tv]
idx_train, idx_val = train_test_split(
    idx_tv, test_size=0.10/0.80, stratify=labels_tv, random_state=SEED)

feat_train = [features[i] for i in idx_train]
feat_val   = [features[i] for i in idx_val]
feat_test  = [features[i] for i in idx_test]

y_train = np.array([f['label'] for f in feat_train])
y_val   = np.array([f['label'] for f in feat_val])
y_test  = np.array([f['label'] for f in feat_test])

print(f'Train: {len(feat_train):,} | Val: {len(feat_val):,} | Test: {len(feat_test):,}')
for name, lbl in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    print(f'  {name} → Normal: {(lbl==0).sum():,}  Anomaly: {(lbl==1).sum():,}  ({100*lbl.mean():.1f}% anomaly)')


# ── Weighted sampler ─────────────────────────────────────────────────────────
class_counts  = np.bincount(y_train)
class_weights = 1.0 / class_counts
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.double),
    num_samples=len(feat_train), replacement=True)

ds_train = HDFSDataset(feat_train)
ds_val   = HDFSDataset(feat_val)
ds_test  = HDFSDataset(feat_test)

dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=0, pin_memory=False)
dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False,    num_workers=0)
dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False,    num_workers=0)

# ── Flat arrays for baselines ─────────────────────────────────────────────────
X_train_seq    = np.array([f['event_ids']    for f in feat_train])
X_val_seq      = np.array([f['event_ids']    for f in feat_val])
X_test_seq     = np.array([f['event_ids']    for f in feat_test])
X_train_struct = np.array([f['struct_feats'] for f in feat_train])
X_val_struct   = np.array([f['struct_feats'] for f in feat_val])
X_test_struct  = np.array([f['struct_feats'] for f in feat_test])

test_missing_alloc = np.array([f['missing_alloc'] for f in feat_test])
test_repl_neq3     = np.array([f['repl_neq3']     for f in feat_test])
val_missing_alloc  = np.array([f['missing_alloc'] for f in feat_val])
val_repl_neq3      = np.array([f['repl_neq3']     for f in feat_val])

print(f'DataLoaders ready. Batches/epoch (train): {len(dl_train)}')


# ── Save everything to cache ──────────────────────────────────────────────────
cache_payload = {
    'feat_train': feat_train, 'feat_val': feat_val, 'feat_test': feat_test,
    'template2id': template2id, 'top5_ids': top5_ids,
    'VOCAB_SIZE': VOCAB_SIZE, 'thread_max': thread_max,
    'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
    'X_train_seq': X_train_seq, 'X_val_seq': X_val_seq, 'X_test_seq': X_test_seq,
    'X_train_struct': X_train_struct, 'X_val_struct': X_val_struct, 'X_test_struct': X_test_struct,
    'test_missing_alloc': test_missing_alloc, 'test_repl_neq3': test_repl_neq3,
    'val_missing_alloc': val_missing_alloc,   'val_repl_neq3': val_repl_neq3,
    'MAX_LEN': MAX_LEN, 'BATCH_SIZE': BATCH_SIZE,
    'MAX_EPOCHS': MAX_EPOCHS, 'LR': LR,
    'WEIGHT_DECAY': WEIGHT_DECAY, 'PATIENCE': PATIENCE,
    'SEED': SEED, 'OUTPUT_DIR': OUTPUT_DIR,
    'CACHE_DIR': CACHE_DIR, 'FIGURE_DIR': FIGURE_DIR, 'MODEL_DIR': MODEL_DIR,
}

cache_path = os.path.join(CACHE_DIR, 'features.pkl')
with open(cache_path, 'wb') as f:
    pickle.dump(cache_payload, f)

print(f'\n✅ Cache saved: {cache_path}')
print(f'   Size: {os.path.getsize(cache_path)/1e6:.1f} MB')
print('\n✅ Notebook 1 complete — run nb2_baselines.ipynb next')


#!/usr/bin/env python3
# =============================================================================
# data_preparation.py
# Notebook 1 of 7 — Log Dataset Tokenisation & Sequence Generation
# =============================================================================
# Purpose:
#   1. Builds a unified Drain tokenizer (vocab size ~9,707) from the full HDFS,
#      BGL, and Spirit datasets.
#   2. Parses raw CSVs into padded int32 sequences of length MAX_SEQ_LEN=64.
#   3. Generates 80/20 train/test splits:
#        - HDFS : stratified random split by BlockId session
#        - BGL  : temporal split (first 80% chronological)
#        - Spirit: temporal split (first 80% chronological)
#   4. Saves output files for downstream training notebooks:
#        tokenizer/drain_tokenizer.json
#        data/{hdfs,bgl,spirit}_sequences.pkl
# =============================================================================
# On Kaggle:
#   - Attach dataset "pfe-log-anomaly" (contains HDFS/BGL/Spirit_Drain.csv)
#   - Internet OFF
#   - GPU: not required (CPU is fine)
#   - Runtime: ~10-15 min on Kaggle CPU
# Downstream notebooks read from this notebook's output dataset.
# =============================================================================

# CELL 1 — Environment Configuration & Imports
import os, gc, json, pickle, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

KAGGLE   = os.path.exists('/kaggle/working')
BASE_IN  = '/kaggle/input/pfe-log-anomaly' if KAGGLE else 'Dataset'
BASE_OUT = '/kaggle/working'               if KAGGLE else 'results/lm_pipeline'

for d in ['tokenizer', 'data', 'models', 'results']:
    os.makedirs(f"{BASE_OUT}/{d}", exist_ok=True)

def find_file(name):
    """Locate a file recursively in input directories."""
    # Direct candidate checks
    candidates = [
        os.path.join(BASE_IN, name),
        os.path.join('/kaggle/input', name),
        os.path.join('Dataset', name),
        os.path.join('..', 'Dataset', name),
        name,
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # Recursive walk fallback
    name_lower = name.lower()
    if os.path.exists('/kaggle/input'):
        root_dir = '/kaggle/input'
    elif os.path.exists(BASE_IN):
        root_dir = BASE_IN
    else:
        root_dir = '.'

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in
                   ['.venv', '.venv311', '.git', '__pycache__', 'node_modules']]
        for f in files:
            if f.lower() == name_lower:
                return os.path.join(root, f)
    raise FileNotFoundError(f"'{name}' not found. Searched in {root_dir}. Please ensure the dataset is attached.")

print(f"[INFO] Environment : {'Kaggle' if KAGGLE else 'Local'}")
print(f"[INFO] Input path  : {BASE_IN}")
print(f"[INFO] Output path : {BASE_OUT}")

# CELL 2 — Build Unified Drain Tokenizer
# Reads the template column from all three CSVs in 5M-row chunks.
# Vocabulary: PAD=0  UNK=1  BOS=2  EOS=3  templates=4+  (sorted alphabetically)
print("\n[CELL 2] Building unified Drain tokenizer ...")
t0 = time.time()

CHUNK = 5_000_000
all_templates = set()

for csv_name in ['HDFS_Drain.csv', 'BGL_Drain.csv', 'Spirit_Drain.csv']:
    path = find_file(csv_name)
    print(f"  Scanning {csv_name} ...")
    for chunk in pd.read_csv(path, usecols=['template'], chunksize=CHUNK,
                             on_bad_lines='skip', low_memory=False):
        all_templates.update(
            chunk['template'].fillna('').astype(str).unique()
        )
        del chunk

sorted_templates = sorted(all_templates)
SPECIAL = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
vocab = {**SPECIAL}
for i, t in enumerate(sorted_templates):
    vocab[t] = i + 4

VOCAB_SIZE = len(vocab)
id_to_template = {str(v): k for k, v in vocab.items()}

tok_path = f"{BASE_OUT}/tokenizer/drain_tokenizer.json"
with open(tok_path, 'w', encoding='utf-8') as f:
    json.dump({
        'vocab': vocab,
        'id_to_template': id_to_template,
        'special_tokens': {
            'pad_token': '<PAD>', 'unk_token': '<UNK>',
            'bos_token': '<BOS>', 'eos_token': '<EOS>',
            'pad_token_id': 0, 'unk_token_id': 1,
            'bos_token_id': 2, 'eos_token_id': 3,
        },
        'stats': {
            'vocab_size': VOCAB_SIZE,
            'unique_templates': len(sorted_templates),
        }
    }, f, indent=2, ensure_ascii=False)

print(f"[OK] Tokenizer saved : {tok_path}")
print(f"     Vocab size       : {VOCAB_SIZE:,}  ({len(sorted_templates):,} templates + 4 special)")
print(f"     Time             : {time.time()-t0:.1f}s")

# CELL 3 — HDFS Sessions (stratified 80/20 random split)
# Groups log lines by BlockId extracted via regex from the 'log' column.
# A session is anomalous if ANY of its lines has Label != 'Normal'.
# Uses pandas groupby.agg for speed (avoids row-by-row Python loop).
print("\n[CELL 3] Processing HDFS sessions ...")
t0 = time.time()

MAX_SEQ_LEN = 64

hdfs_path = find_file('HDFS_Drain.csv')
df = pd.read_csv(hdfs_path, usecols=['log', 'Label', 'template'],
                 on_bad_lines='skip', low_memory=False)
print(f"  Loaded {len(df):,} HDFS rows")

# Extract BlockId (vectorised regex — no per-row Python loop)
df['_bid'] = df['log'].str.extract(r'(blk_-?\d+)', expand=False)
df = df.dropna(subset=['_bid'])

df['_anom'] = (df['Label'].astype(str).str.strip() != 'Normal').astype(np.int8)

# Map template -> id using pandas Series.map (vectorised, ~10x faster than list comprehension)
tmpl_series = df['template'].fillna('').astype(str)
df['_tid']  = tmpl_series.map(vocab).fillna(1).astype(np.int32)
del tmpl_series

print(f"  Grouping {df['_bid'].nunique():,} unique BlockIds ...")
grouped = df.groupby('_bid', sort=False).agg({'_tid': list, '_anom': 'max'})
del df; gc.collect()

block_order  = grouped.index.tolist()
block_events = grouped['_tid'].to_dict()
block_labels = grouped['_anom'].to_dict()
del grouped

n_blocks   = len(block_order)
sequences  = np.zeros((n_blocks, MAX_SEQ_LEN), dtype=np.int32)
labels_arr = np.zeros(n_blocks,               dtype=np.int32)

for i, bid in enumerate(block_order):
    ids     = block_events[bid]
    seq_len = min(len(ids), MAX_SEQ_LEN)
    sequences[i, :seq_len] = ids[:seq_len]
    labels_arr[i]          = block_labels[bid]

del block_events, block_labels, block_order; gc.collect()

X_tr, X_te, y_tr, y_te = train_test_split(
    sequences, labels_arr, test_size=0.20, stratify=labels_arr, random_state=42)
del sequences, labels_arr; gc.collect()

dest = f"{BASE_OUT}/data/hdfs_sequences.pkl"
with open(dest, 'wb') as f:
    pickle.dump({'X_train': X_tr, 'X_test': X_te,
                 'y_train': y_tr, 'y_test':  y_te,
                 'metadata': {
                     'dataset': 'HDFS', 'grouping': 'BlockId sessions',
                     'split': '80/20 stratified random',
                     'max_seq_len': MAX_SEQ_LEN, 'vocab_size': VOCAB_SIZE}},
                f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"[OK] HDFS pickle saved : {dest}")
print(f"     Train {X_tr.shape}  anom={np.mean(y_tr)*100:.2f}%")
print(f"     Test  {X_te.shape}  anom={np.mean(y_te)*100:.2f}%")
print(f"     Time  : {time.time()-t0:.1f}s")
del X_tr, X_te, y_tr, y_te; gc.collect()

# CELL 4 — BGL Windows (temporal 80/20 split)
# Sliding window W=20 S=10 over the full BGL log sequence (sorted chronologically).
# Window label = max(line labels inside window)  i.e. 1 if ANY line is anomalous.
# Uses pandas Series.map for vectorised template->id conversion.
print("\n[CELL 4] Processing BGL windows ...")
t0 = time.time()

W, S = 20, 10
bgl_path = find_file('BGL_Drain.csv')

all_ids    = []
all_labels = []

for chunk in pd.read_csv(bgl_path, usecols=['template', 'label'],
                         chunksize=CHUNK, on_bad_lines='skip', low_memory=False):
    chunk['template'] = chunk['template'].fillna('').astype(str)
    chunk['label']    = chunk['label'].fillna('-').astype(str).str.strip()
    # Vectorised map
    all_ids.extend(chunk['template'].map(vocab).fillna(1).astype(np.int32).tolist())
    all_labels.extend((chunk['label'] != '-').astype(np.int8).tolist())
    del chunk; gc.collect()

event_ids = np.array(all_ids, dtype=np.int32);  del all_ids
label_arr = np.array(all_labels, dtype=np.int8); del all_labels
gc.collect()

n_lines   = len(event_ids)
n_windows = (n_lines - W) // S + 1
print(f"  BGL lines: {n_lines:,}  line-anom: {np.mean(label_arr)*100:.2f}%")
print(f"  Building {n_windows:,} windows ...")

sequences  = np.zeros((n_windows, MAX_SEQ_LEN), dtype=np.int32)
labels_arr = np.zeros(n_windows,               dtype=np.int8)

for i in range(n_windows):
    st = i * S
    sequences[i, :W] = event_ids[st:st+W]
    labels_arr[i]    = label_arr[st:st+W].max()

del event_ids, label_arr; gc.collect()

split    = int(n_windows * 0.8)
X_tr, X_te = sequences[:split], sequences[split:]
y_tr, y_te = labels_arr[:split], labels_arr[split:]
del sequences, labels_arr; gc.collect()

dest = f"{BASE_OUT}/data/bgl_sequences.pkl"
with open(dest, 'wb') as f:
    pickle.dump({'X_train': X_tr, 'X_test': X_te,
                 'y_train': y_tr.astype(np.int32), 'y_test': y_te.astype(np.int32),
                 'metadata': {
                     'dataset': 'BGL', 'grouping': f'Sliding W={W} S={S}',
                     'split': '80/20 temporal',
                     'max_seq_len': MAX_SEQ_LEN, 'window_size': W,
                     'step_size': S, 'vocab_size': VOCAB_SIZE}},
                f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"[OK] BGL pickle saved  : {dest}")
print(f"     Train {X_tr.shape}  anom={np.mean(y_tr)*100:.2f}%")
print(f"     Test  {X_te.shape}  anom={np.mean(y_te)*100:.2f}%")
print(f"     Time  : {time.time()-t0:.1f}s")
del X_tr, X_te, y_tr, y_te; gc.collect()

# CELL 5 — Spirit Windows (temporal 80/20 split)
# Identical pipeline to BGL but applied to Spirit_Drain.csv.
print("\n[CELL 5] Processing Spirit windows ...")
t0 = time.time()

spirit_path = find_file('Spirit_Drain.csv')

all_ids    = []
all_labels = []

for chunk in pd.read_csv(spirit_path, usecols=['template', 'label'],
                         chunksize=CHUNK, on_bad_lines='skip', low_memory=False):
    chunk['template'] = chunk['template'].fillna('').astype(str)
    chunk['label']    = chunk['label'].fillna('-').astype(str).str.strip()
    all_ids.extend(chunk['template'].map(vocab).fillna(1).astype(np.int32).tolist())
    all_labels.extend((chunk['label'] != '-').astype(np.int8).tolist())
    del chunk; gc.collect()

event_ids = np.array(all_ids, dtype=np.int32);  del all_ids
label_arr = np.array(all_labels, dtype=np.int8); del all_labels
gc.collect()

n_lines   = len(event_ids)
n_windows = (n_lines - W) // S + 1
print(f"  Spirit lines: {n_lines:,}  line-anom: {np.mean(label_arr)*100:.2f}%")
print(f"  Building {n_windows:,} windows ...")

sequences  = np.zeros((n_windows, MAX_SEQ_LEN), dtype=np.int32)
labels_arr = np.zeros(n_windows,               dtype=np.int8)

for i in range(n_windows):
    st = i * S
    sequences[i, :W] = event_ids[st:st+W]
    labels_arr[i]    = label_arr[st:st+W].max()

del event_ids, label_arr; gc.collect()

split    = int(n_windows * 0.8)
X_tr, X_te = sequences[:split], sequences[split:]
y_tr, y_te = labels_arr[:split], labels_arr[split:]
del sequences, labels_arr; gc.collect()

dest = f"{BASE_OUT}/data/spirit_sequences.pkl"
with open(dest, 'wb') as f:
    pickle.dump({'X_train': X_tr, 'X_test': X_te,
                 'y_train': y_tr.astype(np.int32), 'y_test': y_te.astype(np.int32),
                 'metadata': {
                     'dataset': 'Spirit', 'grouping': f'Sliding W={W} S={S}',
                     'split': '80/20 temporal',
                     'max_seq_len': MAX_SEQ_LEN, 'window_size': W,
                     'step_size': S, 'vocab_size': VOCAB_SIZE}},
                f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"[OK] Spirit pickle saved : {dest}")
print(f"     Train {X_tr.shape}  anom={np.mean(y_tr)*100:.2f}%")
print(f"     Test  {X_te.shape}  anom={np.mean(y_te)*100:.2f}%")
print(f"     Time  : {time.time()-t0:.1f}s")
del X_tr, X_te, y_tr, y_te; gc.collect()

# CELL 6 — Verification Summary
print("\n[CELL 6] Verification summary")
print(f"  Vocab size : {VOCAB_SIZE:,}")
for name in ['hdfs', 'bgl', 'spirit']:
    with open(f"{BASE_OUT}/data/{name}_sequences.pkl", 'rb') as f:
        d = pickle.load(f)
    print(f"  {name.upper():6s}  train={d['X_train'].shape}  anom={np.mean(d['y_train'])*100:.2f}%"
          f"  |  test={d['X_test'].shape}  anom={np.mean(d['y_test'])*100:.2f}%")
print("[DONE] data_preparation complete.")

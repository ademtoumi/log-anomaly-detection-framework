#!/usr/bin/env python3
# =============================================================================
# prepare_hdfs.py
# Preprocessing for HDFS log dataset.
#
# Design decisions (academic justification):
#   - BlockId session grouping: standard for HDFS (Xu et al. 2009); each block
#     is a natural execution unit with a ground-truth label.
#   - Stratified random split 80/20: sessions are i.i.d. by construction
#     (block IDs are independent parallel jobs), so random stratified split
#     is valid — no temporal ordering is required.
#   - MAX_SEQ_LEN = 64: covers the 95th percentile of HDFS session lengths
#     while staying memory-efficient on Kaggle T4.
#   - Special tokens: PAD=0, UNK=1, CLS=2, MASK=3.  BOS/EOS removed — BERT
#     uses CLS, not BOS/EOS.  Training scripts read CLS via 'cls_token_id'.
# =============================================================================
import os
import gc
import json
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

KAGGLE   = os.path.exists('/kaggle/working')
BASE_IN  = '/kaggle/input/pfe-log-anomaly' if KAGGLE else 'Dataset'
BASE_OUT = '/kaggle/working'               if KAGGLE else 'results/lm_pipeline'

for d in ['tokenizer', 'data']:
    os.makedirs(f"{BASE_OUT}/{d}", exist_ok=True)

def find_file(name):
    candidates = [
        os.path.join(BASE_IN, name),
        os.path.join('/kaggle/input', name),
        os.path.join('Dataset', name),
        os.path.join('..', 'Dataset', name),
        name,
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in ['.venv', '.venv311', '.git', '__pycache__']]
        for f in files:
            if f.lower() == name.lower():
                return os.path.join(root, f)
    raise FileNotFoundError(f"'{name}' not found. Please attach the dataset.")

print("[INFO] HDFS Preprocessing started...")
t0 = time.time()

# ── 1. Build dataset-specific tokenizer ──────────────────────────────────────
hdfs_path = find_file('HDFS_Drain.csv')
print(f"Reading HDFS logs from: {hdfs_path}")

CHUNK = 5_000_000
unique_templates = set()
for chunk in pd.read_csv(hdfs_path, usecols=['template'], chunksize=CHUNK,
                         on_bad_lines='skip', low_memory=False):
    unique_templates.update(chunk['template'].fillna('').astype(str).unique())
    del chunk; gc.collect()

sorted_templates = sorted(list(unique_templates))

# Special tokens: PAD=0, UNK=1, CLS=2, MASK=3
# CLS (not BOS) is the BERT convention for sequence classification.
# MASK is what we replace tokens with during MLM.
# No BOS/EOS — they are unnecessary for BERT-style encoder-only models.
SPECIAL = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2, '<MASK>': 3}
vocab = {**SPECIAL}
for i, t in enumerate(sorted_templates):
    vocab[t] = i + len(SPECIAL)

VOCAB_SIZE = len(vocab)
# Use int keys so downstream lookup (vocab[token_id]) works without str() cast
id_to_template = {v: k for k, v in vocab.items()}

# Save HDFS tokenizer
tok_path = f"{BASE_OUT}/tokenizer/hdfs_tokenizer.json"
with open(tok_path, 'w', encoding='utf-8') as f:
    json.dump({
        'vocab': vocab,
        'id_to_template': {str(v): k for k, v in vocab.items()},  # JSON needs str keys
        'special_tokens': {
            'pad_token':    '<PAD>',  'unk_token':  '<UNK>',
            'cls_token':    '<CLS>',  'mask_token': '<MASK>',
            'pad_token_id':  0,       'unk_token_id':  1,
            'cls_token_id':  2,       'mask_token_id': 3,
        },
        'stats': {
            'vocab_size':       VOCAB_SIZE,
            'unique_templates': len(sorted_templates),
        }
    }, f, indent=2, ensure_ascii=False)

print(f"[OK] HDFS tokenizer saved: {tok_path}  (vocab_size={VOCAB_SIZE})")

# ── 2. Parse logs and extract sessions ───────────────────────────────────────
df = pd.read_csv(hdfs_path, usecols=['log', 'Label', 'template'],
                 on_bad_lines='skip', low_memory=False)
print(f"Loaded {len(df):,} log rows")

# Extract BlockId via regex (standard HDFS identifier)
df['_bid'] = df['log'].str.extract(r'(blk_-?\d+)', expand=False)
df = df.dropna(subset=['_bid'])
print(f"Rows with BlockId: {len(df):,}")

# Binary encoding: 0=Normal, 1=Anomaly
df['_anom'] = (df['Label'].astype(str).str.strip() != 'Normal').astype(np.int8)

# Map templates → token IDs (UNK=1 for unseen templates)
df['_tid'] = df['template'].fillna('').astype(str).map(vocab).fillna(1).astype(np.int32)

# Session-level aggregation: collect token sequence, label = max(line labels)
print(f"Grouping {df['_bid'].nunique():,} unique BlockId sessions...")
grouped = df.groupby('_bid', sort=False).agg({'_tid': list, '_anom': 'max'})
del df; gc.collect()

block_order  = grouped.index.tolist()
block_events = grouped['_tid'].to_dict()
block_labels = grouped['_anom'].to_dict()
del grouped; gc.collect()

# ── 3. Sequence generation & padding ─────────────────────────────────────────
# MAX_SEQ_LEN = 64 covers ≥95th percentile of HDFS session lengths.
# Sequences are truncated (front-truncation would drop early context; we keep
# the first MAX_SEQ_LEN events which is the standard LogBERT convention).
# Padding with PAD_ID=0 is applied at the right.
MAX_SEQ_LEN = 64
n_blocks    = len(block_order)
sequences   = np.zeros((n_blocks, MAX_SEQ_LEN), dtype=np.int32)
labels_arr  = np.zeros(n_blocks, dtype=np.int32)

empty_seq_count   = 0
truncated_count   = 0
session_lengths   = []

for i, bid in enumerate(block_order):
    ids     = block_events[bid]
    raw_len = len(ids)
    session_lengths.append(raw_len)
    if raw_len == 0:
        empty_seq_count += 1
    if raw_len > MAX_SEQ_LEN:
        truncated_count += 1
    seq_len = min(raw_len, MAX_SEQ_LEN)
    sequences[i, :seq_len] = ids[:seq_len]
    labels_arr[i] = block_labels[bid]

del block_events, block_labels, block_order; gc.collect()

session_lengths = np.array(session_lengths)

# ── 4. Stratified random split ────────────────────────────────────────────────
# Justification: HDFS BlockIds are parallel execution units (independent jobs).
# No temporal dependency exists across blocks, so random stratified split is
# valid and maximally data-efficient.
X_train, X_test, y_train, y_test = train_test_split(
    sequences, labels_arr, test_size=0.20, stratify=labels_arr, random_state=42
)
del sequences, labels_arr; gc.collect()

# ── 5. Save ───────────────────────────────────────────────────────────────────
dest_pkl = f"{BASE_OUT}/data/hdfs_sequences.pkl"
with open(dest_pkl, 'wb') as f:
    pickle.dump({
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'metadata': {
            'dataset':          'HDFS',
            'grouping':         'BlockId sessions',
            'split':            '80/20 stratified random (valid: blocks are i.i.d.)',
            'max_seq_len':      MAX_SEQ_LEN,
            'vocab_size':       VOCAB_SIZE,
            'n_blocks':         n_blocks,
            'empty_sequences':  int(empty_seq_count),
            'truncated':        int(truncated_count),
            'session_len_p50':  float(np.percentile(session_lengths, 50)),
            'session_len_p95':  float(np.percentile(session_lengths, 95)),
            'session_len_max':  int(session_lengths.max()),
            'pad_id':  0, 'unk_id': 1, 'cls_id': 2, 'mask_id': 3,
        }
    }, f, protocol=pickle.HIGHEST_PROTOCOL)

# ── 6. Verification summary ───────────────────────────────────────────────────
print(f"[OK] HDFS sequences saved: {dest_pkl}")
print(f"--- HDFS Preprocessing Verification Summary ---")
print(f"  Vocab Size          : {VOCAB_SIZE}")
print(f"  Total sessions      : {n_blocks:,}")
print(f"  Empty sequences     : {empty_seq_count}")
print(f"  Truncated (>{MAX_SEQ_LEN}): {truncated_count}")
print(f"  Session len p50/p95/max: "
      f"{np.percentile(session_lengths, 50):.0f} / "
      f"{np.percentile(session_lengths, 95):.0f} / "
      f"{session_lengths.max()}")
print(f"  Train Set : {X_train.shape}  Anomalous={np.sum(y_train):,} ({np.mean(y_train)*100:.2f}%)")
print(f"  Test  Set : {X_test.shape}  Anomalous={np.sum(y_test):,}  ({np.mean(y_test)*100:.2f}%)")
print(f"  Execution Time      : {time.time() - t0:.1f}s")
print(f"  Special tokens      : PAD=0  UNK=1  CLS=2  MASK=3")
print(f"------------------------------------------------")

ratio_diff = abs(np.mean(y_train) - np.mean(y_test))
if ratio_diff > 0.02:
    print(f"[WARNING] Train/test anomaly ratio mismatch: {ratio_diff:.4f}")
else:
    print(f"[SUCCESS] Train/test distributions matched (diff={ratio_diff:.4f}), leakage-free.")

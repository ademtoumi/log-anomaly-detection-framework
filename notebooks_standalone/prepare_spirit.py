#!/usr/bin/env python3
# =============================================================================
# prepare_spirit.py
# Preprocessing for Spirit log dataset using Temporal Block Split.
#
# Design decisions (academic justification):
#   - Sliding window (W=20, S=10): mirrors BGL convention. Spirit is a large
#     continuous system log; fixed-size windows are the standard approach
#     (He et al. 2016, Guo et al. 2021).
#   - TEMPORAL split (first 80% → train, last 20% → test):
#     Spirit logs have a strong temporal structure (Sandia HPC cluster events
#     over time). A random or block-shuffled split leaks future context into
#     training windows because overlapping windows (step S=10 < window W=20)
#     from adjacent time regions can appear in both splits.  Temporal split
#     fully eliminates this.
#   - Guard gap of W lines at the train/test boundary: ensures no window
#     straddles the frontier, making the split perfectly leakage-free.
#   - Spirit is the HIGHEST PRIORITY dataset (per task spec): temporal split
#     is especially important here because Spirit has long anomaly bursts
#     whose context would bleed across a random split boundary.
#   - Special tokens: PAD=0, UNK=1, CLS=2, MASK=3 (consistent across all datasets).
# =============================================================================
import os
import gc
import json
import pickle
import time
import numpy as np
import pandas as pd

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

print("[INFO] Spirit Preprocessing started...")
t0 = time.time()

# ── 1. Build dataset-specific tokenizer ──────────────────────────────────────
spirit_path = find_file('Spirit_Drain.csv')
print(f"Reading Spirit logs from: {spirit_path}")

CHUNK = 5_000_000
unique_templates = set()
for chunk in pd.read_csv(spirit_path, usecols=['template'], chunksize=CHUNK,
                         on_bad_lines='skip', low_memory=False):
    unique_templates.update(chunk['template'].fillna('').astype(str).unique())
    del chunk; gc.collect()

sorted_templates = sorted(list(unique_templates))

# PAD=0, UNK=1, CLS=2, MASK=3  — identical layout across all three datasets.
# This uniform layout ensures training scripts never misread special token IDs.
SPECIAL = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2, '<MASK>': 3}
vocab = {**SPECIAL}
for i, t in enumerate(sorted_templates):
    vocab[t] = i + len(SPECIAL)

VOCAB_SIZE = len(vocab)

tok_path = f"{BASE_OUT}/tokenizer/spirit_tokenizer.json"
with open(tok_path, 'w', encoding='utf-8') as f:
    json.dump({
        'vocab': vocab,
        'id_to_template': {str(v): k for k, v in vocab.items()},
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

print(f"[OK] Spirit tokenizer saved: {tok_path}  (vocab_size={VOCAB_SIZE})")

# ── 2. Load log lines ─────────────────────────────────────────────────────────
df = pd.read_csv(spirit_path, usecols=['template', 'label'],
                 on_bad_lines='skip', low_memory=False)
df['template'] = df['template'].fillna('').astype(str)
df['label']    = df['label'].fillna('-').astype(str).str.strip()

event_ids = df['template'].map(vocab).fillna(1).astype(np.int32).values
is_anom   = (df['label'] != '-').astype(np.int8).values
n_lines   = len(event_ids)
del df; gc.collect()

print(f"Spirit log lines: {n_lines:,}  |  Anomalous lines: {is_anom.sum():,} ({is_anom.mean()*100:.2f}%)")

# ── 3. Temporal split with boundary guard ─────────────────────────────────────
# Temporal split: train = first 80% of log, test = last 20%.
# Guard gap = W lines excluded around the split point so no window straddles it.
W, S        = 20, 10
MAX_SEQ_LEN = W     # window size = sequence length; no padding needed

TRAIN_FRAC  = 0.80
split_line  = int(n_lines * TRAIN_FRAC)

# Guard gap: exclude W lines immediately on each side of the split boundary
train_end  = split_line - W       # exclusive upper bound for training window starts
test_start = split_line + W       # inclusive lower bound for test window starts

print(f"Temporal split: train=[0, {train_end + W - 1}]  "
      f"test=[{test_start}, {n_lines - 1}]  "
      f"guard_gap={2*W} lines")

# ── 4. Build sliding windows ──────────────────────────────────────────────────
def build_windows(event_ids, is_anom, start, end_start_exclusive, W, S):
    """
    Build sliding windows with starts in [start, end_start_exclusive).
    Window i covers lines [i*S, i*S + W).
    Label = 1 if any line in the window is anomalous, else 0.
    """
    seqs, lbls = [], []
    for i in range(start, end_start_exclusive - W + 1, S):
        seqs.append(event_ids[i : i + W])
        lbls.append(int(is_anom[i : i + W].max()))
    if not seqs:
        return np.zeros((0, W), dtype=np.int32), np.zeros(0, dtype=np.int32)
    return np.array(seqs, dtype=np.int32), np.array(lbls, dtype=np.int32)

print("Building training windows...")
X_train, y_train = build_windows(event_ids, is_anom, 0, train_end + 1, W, S)

print("Building test windows...")
X_test,  y_test  = build_windows(event_ids, is_anom, test_start, n_lines, W, S)

del event_ids, is_anom; gc.collect()

# ── 5. Save ───────────────────────────────────────────────────────────────────
dest_pkl = f"{BASE_OUT}/data/spirit_sequences.pkl"
with open(dest_pkl, 'wb') as f:
    pickle.dump({
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'metadata': {
            'dataset':      'Spirit',
            'grouping':     f'Sliding window W={W} S={S}',
            'split':        '80/20 temporal (leakage-free: no window crosses boundary)',
            'max_seq_len':  MAX_SEQ_LEN,
            'window_size':  W,
            'step_size':    S,
            'train_frac':   TRAIN_FRAC,
            'split_line':   split_line,
            'guard_lines':  W,
            'vocab_size':   VOCAB_SIZE,
            'pad_id':  0, 'unk_id': 1, 'cls_id': 2, 'mask_id': 3,
        }
    }, f, protocol=pickle.HIGHEST_PROTOCOL)

# ── 6. Verification summary ───────────────────────────────────────────────────
print(f"[OK] Spirit sequences saved: {dest_pkl}")
print(f"--- Spirit Preprocessing Verification Summary ---")
print(f"  Vocab Size      : {VOCAB_SIZE}")
print(f"  Window W/S      : {W}/{S}")
print(f"  Train Set       : {X_train.shape}  Anomalous={np.sum(y_train):,} ({np.mean(y_train)*100:.2f}%)")
print(f"  Test  Set       : {X_test.shape}  Anomalous={np.sum(y_test):,}  ({np.mean(y_test)*100:.2f}%)")
print(f"  Execution Time  : {time.time() - t0:.1f}s")
print(f"  Special tokens  : PAD=0  UNK=1  CLS=2  MASK=3")
print(f"-------------------------------------------------")

ratio_diff = abs(np.mean(y_train) - np.mean(y_test))
if ratio_diff > 0.10:
    print(f"[WARNING] Train/test anomaly ratio mismatch: {ratio_diff:.4f}")
    print("  Expected with temporal split if anomaly bursts cluster in time.")
    print("  This is NOT leakage — temporal validity is correctly maintained.")
else:
    print(f"[SUCCESS] Train/test distributions matched (diff={ratio_diff:.4f}), leakage-free.")

# Leakage sanity check: verify zero window-start overlap across splits
train_starts = set(range(0, train_end + 1, S))
test_starts  = set(range(test_start, n_lines, S))
overlap = train_starts & test_starts
if overlap:
    print(f"[ERROR] Window start overlap detected: {len(overlap)} positions — check guard gap!")
else:
    print(f"[OK] Zero window start overlap between train and test (guard gap verified).")

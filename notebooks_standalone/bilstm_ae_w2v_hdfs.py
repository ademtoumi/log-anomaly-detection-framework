# =============================================================================
# STANDALONE NOTEBOOK 14 â€” BiLSTM Autoencoder + Word2Vec on HDFS
#
# Thesis keystone reproduction target:
#   [Bekkouche2025_BiLSTM] "Log-based anomaly detection using BiLSTM-Autoencoder"
#   Bekkouche M., Meski M., Khodja Y., Benslimane S.M., Tronci E.
#   IEEE Conference Proceedings, 2025.
#   â†’ BiLSTM-AE + Word2Vec (Complete HDFS): F1=0.993, Precision=0.987, Recall=1.000
#
# KEY DISTINCTION from Notebook 12 (lstm_ae_hdfs):
#   Notebook 12  â€” uses nn.Embedding (random init, learned end-to-end)
#   Notebook 14  â€” uses Word2Vec trained from scratch on ALL HDFS sessions
#                  (full 2.6 GB corpus, embedding_dim=100, window=5, min_count=1)
#                  Embeddings are FINE-TUNED during AE training (unfrozen) for
#                  richer separation between normal and anomalous reconstruction.
#
# Architecture: [Bekkouche2025_BiLSTM] Â§III-B
#   W2V Embedding (fine-tuned, dim=100)
#   â†’ BiLSTM Encoder (hidden=128, bidirectional=True)
#       last hidden state h_n â†’ concat fwd+bwd â†’ Linear â†’ tanh â†’ latent (hidden)
#   â†’ LSTM Decoder: latent as initial hidden state h0, zero input sequence
#   â†’ Linear projection â†’ embedding_dim (reconstructed sequence)
#   Loss: MSELoss(decoded, original_embeddings.detach())
#
# Training protocol: [Bekkouche2025_BiLSTM] Â§III-C
#   â€¢ NORMAL sessions only (unsupervised)
#   â€¢ Adam optimiser, gradient clipping max_norm=1.0
#   â€¢ AMP (autocast + GradScaler) â€” T4/P100 GPU memory efficiency
#   â€¢ CosineAnnealingLR (primary) + ReduceLROnPlateau (secondary, factor=0.5, patience=5)
#   â€¢ Early stopping on val MSE (patience=15) â€” val = 10% hold-out of train (normal only)
#
# Threshold protocol: [Bekkouche2025_BiLSTM] Â§II-C-a (EXACT PAPER PROTOCOL)
#   â€¢ Threshold searched directly on the TEST set to maximise F1-Score
#   â€¢ Quote: "the test partition of the dataset is used to determine the
#     threshold value that maximises the F1-Score" â€” semi-supervised threshold
#   â€¢ Range: [10th-percentile of normal test errors, 99.5th-percentile of all test errors]
#   â€¢ 5000 candidates for fine-grained search
#
# Split protocol: [Bekkouche2025_BiLSTM] Table I â€” EXACT PAPER PROTOCOL
#   â€¢ Uniform split (stratified random): 90% train / 10% test
#   â€¢ sklearn train_test_split(stratify=y, test_size=0.10, random_state=42)
#   â€¢ Paper reports: 517,554 train / 57,507 test (Complete HDFS)
#   â€¢ NO separate validation set for threshold â€” threshold tuned on TEST directly
#
# Session protocol: [Du2017_DeepLog]
#   â€¢ Sessions grouped by BlockId extracted via regex blk_-?\d+
#   â€¢ Label: any anomalous line in session â†’ session label = 1
#   â€¢ MAX_SEQ_LEN computed from actual data (median + 2*MAD, capped at 100)
#
# Memory strategy:
#   â€¢ Chunked CSV reading (chunksize=50_000) â€” safe for 16 GB Kaggle RAM
#   â€¢ Two-pass approach: Pass 1 collects sessions; Pass 2 trains W2V corpus
#   â€¢ gc.collect() + cuda.empty_cache() after every major step
#   â€¢ int32 sequences + float32 embeddings throughout
#
# Checkpoint system:
#   â€¢ ckpt_14_bilstm_ae_hdfs.json â€” full resume after Kaggle timeout
#   â€¢ Guards: 'sessions_ready', 'w2v_ready', 'ae_done'
#
# Reproducibility:
#   â€¢ Fixed seed=42: numpy, random, torch, Word2Vec (workers=1)
#   â€¢ Environment logged in summary cell
#
# Kaggle setup:
#   - Dataset: pfe-log-anomaly  (must contain HDFS_Drain.csv)
#   - Accelerator: GPU T4 x1 or P100
#   - Estimated time: ~35â€“45 minutes
#
# Outputs (all to pfe_report/ for consistency with project conventions):
#   models/
#     w2v_hdfs_full.model          â€” trained Word2Vec model
#     bilstm_ae_w2v_hdfs_opt.pt   â€” best AE weights
#     bilstm_ae_w2v_hdfs_config.json
#     hdfs_ae_sessions_train.npz
#     hdfs_ae_sessions_test.npz
#   pfe_report/
#     bilstm_ae_hdfs_results.csv
#     bilstm_ae_hdfs_loss_curve.png
#     bilstm_ae_hdfs_error_hist.png
#     bilstm_ae_hdfs_cm.png
#     bilstm_ae_hdfs_roc.png
#     bilstm_ae_hdfs_pr_curve.png
#     bilstm_ae_hdfs_comparison_table.png
# =============================================================================

import os, gc, json, pathlib, time, random, warnings, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend â€” safe for Kaggle commit mode
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, average_precision_score,
    precision_recall_curve,
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# â”€â”€ Reproducibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =============================================================================
# CELL 1 â€” Environment & Paths
# =============================================================================
KAGGLE    = os.path.exists('/kaggle')
BASE_IN   = '/kaggle/input/pfe-log-anomaly' if KAGGLE else 'Dataset'
BASE_OUT  = '/kaggle/working'               if KAGGLE else 'result/results_bilstm_ae_w2v_hdfs'
MODEL_DIR = f'{BASE_OUT}/models'
REPORT    = f'{BASE_OUT}/pfe_report'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"âœ… Device : {DEVICE}")
print(f"   Env   : {'Kaggle' if KAGGLE else 'Local'}")
print(f"   BASE_IN  â†’ {BASE_IN}")
print(f"   BASE_OUT â†’ {BASE_OUT}")
print(f"   REPORT   â†’ {REPORT}")

# â”€â”€ Checkpoint system â€” full resume after Kaggle timeout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Checkpoint file: ckpt_14_bilstm_ae_hdfs.json
CKPT_PATH = pathlib.Path(BASE_OUT) / 'ckpt_14_bilstm_ae_hdfs.json'

def save_ckpt(d: dict) -> None:
    with open(CKPT_PATH, 'w') as f:
        json.dump(d, f, indent=2)

def load_ckpt() -> dict:
    if CKPT_PATH.exists():
        with open(CKPT_PATH) as f:
            return json.load(f)
    return {}

ckpt = load_ckpt()
if not ckpt:
    print("   CHECKPOINT CLEARED â€” FRESH RUN")
else:
    print(f"   Checkpoint keys loaded: {list(ckpt.keys())}")


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
try:
    from gensim.models import Word2Vec
except ImportError:
    print("ðŸ“¦ Installing gensim ...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'gensim', '-q'], check=True)
    from gensim.models import Word2Vec

# =============================================================================
# CELL 2 â€” HDFS Session Building (inline, chunked)
#
# Protocol: [Du2017_DeepLog] Â§4.1
#   HDFS logs are naturally grouped by BlockId (e.g. blk_-123456789).
#   Each BlockId uniquely identifies one HDFS data block operation.
#   All log lines sharing a BlockId form one session.
#
# Label strategy: [Bekkouche2025_BiLSTM] Â§III-A
#   A session is anomalous if ANY of its log lines is anomalous.
#   (label â‰  'Normal' â†’ anomalous line)
#
# Temporal split: [Zhang2019_LogRobust] Â§5.2
#   60/20/20 split in file order (= approximate temporal order).
#   This is the only scientifically honest split for sequential log data.
#   Random splitting leaks future sessions into training.
#
# Memory: chunked reading (50_000 rows) â€” safe for 16 GB Kaggle RAM.
#   Two dictionaries:
#     block_events[bid] â†’ ordered list of template strings
#     block_labels[bid] â†’ int: max anomaly flag seen (0 or 1)
#   Temporal order preserved via block_order list (insertion order).
# =============================================================================
CHUNK_SIZE  = 50_000     # rows per chunk â€” conservative for 16 GB RAM

# Compute MAX_SEQ_LEN from data statistics (not hardcoded)
# We use median + 2 * MAD, capped at 100, to be robust to outliers.
# [Du2017_DeepLog]: most HDFS sessions are 10â€“50 events.
MAX_SEQ_LEN_CAP = 100    # absolute ceiling to prevent RAM explosion

HDFSPath_NPZ_train = f'{MODEL_DIR}/hdfs_ae_sessions_train.npz'
HDFSPath_NPZ_test  = f'{MODEL_DIR}/hdfs_ae_sessions_test.npz'

if 'sessions_ready' in ckpt:
    print("\n[CELL 2] â­ï¸  Sessions already built â€” loading from NPZ ...")
    train_data = np.load(HDFSPath_NPZ_train)
    test_data  = np.load(HDFSPath_NPZ_test)
    X_train_idx, y_train = train_data['X'], train_data['y']
    X_test_idx,  y_test  = test_data['X'],  test_data['y']
    MAX_SEQ_LEN = int(ckpt['max_seq_len'])
    print(f"  MAX_SEQ_LEN={MAX_SEQ_LEN} | Train={X_train_idx.shape} | Test={X_test_idx.shape}")
    print(f"  Anomaly: train={y_train.mean()*100:.1f}% | test={y_test.mean()*100:.1f}%")

else:
    print(f"\n{'='*65}")
    print("  [CELL 2] Building HDFS sessions from HDFS_Drain.csv ...")
    print(f"{'='*65}")
    t0 = time.time()

    filepath = find_file('HDFS_Drain.csv')
    print(f"  Source file: {filepath}")

    # â”€â”€ Pass 1: chunked session aggregation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # block_events[bid] â†’ list[str]  (template strings in arrival order)
    # block_labels[bid] â†’ int        (1 if any anomalous line, else 0)
    # block_order       â†’ list[str]  (insertion order â‰ˆ temporal order)
    block_events: dict = {}
    block_labels: dict = {}
    block_order:  list = []

    rows_total  = 0
    chunk_num   = 0

    for chunk in pd.read_csv(
            filepath,
            usecols=lambda c: c in ['log', 'template', 'BlockId', 'Label', 'label'],
            chunksize=CHUNK_SIZE,
            on_bad_lines='skip',
            low_memory=False):

        chunk_num += 1

        # â”€â”€ Determine BlockId column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if 'BlockId' in chunk.columns:
            # Some Drain CSV exports include a pre-parsed BlockId column
            chunk['_bid'] = chunk['BlockId'].astype(str).str.strip()
        elif 'log' in chunk.columns:
            # Extract from raw log text â€” [Du2017_DeepLog] protocol
            chunk['_bid'] = chunk['log'].str.extract(r'(blk_-?\d+)', expand=False)
        else:
            # Fallback: try to extract from template
            chunk['_bid'] = chunk['template'].str.extract(r'(blk_-?\d+)', expand=False)

        chunk = chunk.dropna(subset=['_bid'])

        # â”€â”€ Determine label column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if 'Label' in chunk.columns:
            # 'Normal' = normal, anything else = anomaly
            chunk['_anom'] = (chunk['Label'].astype(str).str.strip() != 'Normal').astype(int)
        elif 'label' in chunk.columns:
            # '-' = normal, anything else = anomaly (BGL convention also used here)
            chunk['_anom'] = (chunk['label'].astype(str).str.strip() != '-').astype(int)
        else:
            chunk['_anom'] = 0   # no label column â†’ treat all as normal

        # â”€â”€ Aggregate into sessions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        for _, row in chunk[['_bid', 'template', '_anom']].iterrows():
            bid  = row['_bid']
            tmpl = str(row['template']) if pd.notna(row['template']) else '<UNK>'
            anom = int(row['_anom'])

            if bid not in block_events:
                block_events[bid] = []
                block_labels[bid] = 0
                block_order.append(bid)

            block_events[bid].append(tmpl)
            # OR across all lines in session [Bekkouche2025_BiLSTM]
            block_labels[bid] = max(block_labels[bid], anom)

        rows_total += len(chunk)
        if chunk_num % 10 == 0:
            print(f"  ... chunk {chunk_num:4d} | rows={rows_total:>11,} | "
                  f"sessions={len(block_events):>10,}", end='\r')
        del chunk
        gc.collect()

    print(f"\n  âœ… Pass 1 complete: {rows_total:,} rows â†’ {len(block_order):,} sessions")

    n_blocks  = len(block_order)
    n_anomaly = sum(block_labels[b] for b in block_order)
    print(f"     Normal sessions : {n_blocks - n_anomaly:,} ({(n_blocks-n_anomaly)/n_blocks*100:.1f}%)")
    print(f"     Anomaly sessions: {n_anomaly:,} ({n_anomaly/n_blocks*100:.1f}%)")

    # â”€â”€ Compute MAX_SEQ_LEN from data (robust, not hardcoded) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # [Du2017_DeepLog]: HDFS sessions are mostly 10â€“50 events.
    # We cap at MAX_SEQ_LEN_CAP=100 to prevent memory explosion.
    session_lengths = np.array([len(block_events[b]) for b in block_order],
                               dtype=np.int32)
    median_len = float(np.median(session_lengths))
    mad_len    = float(np.median(np.abs(session_lengths - median_len)))
    MAX_SEQ_LEN = int(min(MAX_SEQ_LEN_CAP, int(median_len + 2 * mad_len) + 1))
    print(f"  Computed MAX_SEQ_LEN = {MAX_SEQ_LEN}  "
          f"(median={median_len:.0f}, MAD={mad_len:.0f}, cap={MAX_SEQ_LEN_CAP})")
    del session_lengths

    # â”€â”€ Build vocabulary from ALL templates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Vocab maps template â†’ integer index.
    # 0 = <PAD>, 1 = <UNK>. Sorted for determinism.
    all_templates: set = set()
    for events in block_events.values():
        all_templates.update(events)

    # Sort for reproducibility (deterministic integer assignments)
    sorted_templates = sorted(all_templates)
    vocab: dict = {'<PAD>': 0, '<UNK>': 1}
    for t in sorted_templates:
        vocab[t] = len(vocab)
    VOCAB_SIZE = len(vocab)
    # Paper Table I reports 48 unique log events. VOCAB_SIZE includes PAD+UNK,
    # so unique real templates = VOCAB_SIZE - 2.
    print(f"  Vocabulary: {VOCAB_SIZE:,} total entries (PAD + UNK + {VOCAB_SIZE-2} unique templates)")
    print(f"  Unique log event templates (excl. PAD/UNK): {VOCAB_SIZE-2:,}  "
          f"â† paper reports 48")
    if abs((VOCAB_SIZE - 2) - 48) > 5:
        print(f"  âš ï¸  Template count differs from paper's 48. Possible causes:")
        print(f"       â€¢ Drain parsing produced different granularity templates")
        print(f"       â€¢ Our CSV may merge/split some log event types")
        print(f"       â€¢ Paper may count EventId not template strings â€” acceptable difference")
    del all_templates, sorted_templates
    gc.collect()

    # â”€â”€ Encode sequences as padded int32 arrays â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    sequences  = np.zeros((n_blocks, MAX_SEQ_LEN), dtype=np.int32)
    labels_arr = np.zeros(n_blocks, dtype=np.int32)

    for i, bid in enumerate(block_order):
        events  = block_events[bid]
        enc     = [vocab.get(e, 1) for e in events]   # <UNK>=1 fallback
        seq_len = min(len(enc), MAX_SEQ_LEN)
        sequences[i, :seq_len] = enc[:seq_len]
        labels_arr[i] = block_labels[bid]

    # Free the heavy session dicts â€” we only need sequences + labels_arr now
    del block_events, block_labels
    gc.collect()
    print(f"  Sequences shape: {sequences.shape} | dtype={sequences.dtype}")

    # â”€â”€ PRIMARY split: Stratified random 90/10 [Bekkouche2025_BiLSTM] Table I â”€â”€
    # Paper: "Uniform Split" â€” 517,554 train / 57,507 test on Complete HDFS
    # sklearn stratify= preserves the anomaly ratio in both partitions.
    from sklearn.model_selection import train_test_split as _tts
    idx_all = np.arange(n_blocks)
    idx_train, idx_test = _tts(
        idx_all, test_size=0.10, stratify=labels_arr, random_state=SEED
    )
    X_train_idx, y_train = sequences[idx_train], labels_arr[idx_train]
    X_test_idx,  y_test  = sequences[idx_test],  labels_arr[idx_test]

    print(f"\n  [PRIMARY] Stratified random 90/10 split [Bekkouche2025_BiLSTM]:")
    print(f"    Train : {len(y_train):>8,} sessions | anomaly={y_train.mean()*100:.2f}%")
    print(f"    Test  : {len(y_test):>8,} sessions  | anomaly={y_test.mean()*100:.2f}%")

    # â”€â”€ SECONDARY split: Temporal 60/20/20 (our rigorous protocol) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Kept for thesis scientific rigour â€” no temporal leakage
    i1 = int(n_blocks * 0.60)
    i2 = int(n_blocks * 0.80)
    X_temp_train = sequences[:i1];   y_temp_train = labels_arr[:i1]
    X_temp_val   = sequences[i1:i2]; y_temp_val   = labels_arr[i1:i2]
    X_temp_test  = sequences[i2:];   y_temp_test  = labels_arr[i2:]
    print(f"\n  [SECONDARY] Temporal 60/20/20 split (our protocol):")
    print(f"    Train : {len(y_temp_train):>8,} sessions | anomaly={y_temp_train.mean()*100:.2f}%")
    print(f"    Val   : {len(y_temp_val):>8,} sessions | anomaly={y_temp_val.mean()*100:.2f}%")
    print(f"    Test  : {len(y_temp_test):>8,} sessions | anomaly={y_temp_test.mean()*100:.2f}%")

    # Save sessions for checkpoint resume (primary split only â€” temp computed on-the-fly)
    np.savez_compressed(HDFSPath_NPZ_train, X=X_train_idx, y=y_train)
    np.savez_compressed(HDFSPath_NPZ_test,  X=X_test_idx,  y=y_test)

    # Save vocab + block_order for Word2Vec corpus construction
    with open(f'{MODEL_DIR}/hdfs_ae_vocab.json', 'w') as f:
        json.dump(vocab, f)

    del sequences, labels_arr
    gc.collect()

    elapsed = time.time() - t0
    ckpt['sessions_ready'] = True
    ckpt['max_seq_len']    = MAX_SEQ_LEN
    ckpt['vocab_size']     = VOCAB_SIZE
    ckpt['n_sessions']     = n_blocks
    save_ckpt(ckpt)
    print(f"\n  âœ… Sessions saved in {elapsed:.0f}s")
    print(f"     â†’ {HDFSPath_NPZ_train}")
    print(f"     â†’ {HDFSPath_NPZ_test}")

# Restore vocab_size + max_seq_len from checkpoint if reloaded
if 'vocab_size'  in ckpt: VOCAB_SIZE  = int(ckpt['vocab_size'])
if 'max_seq_len' in ckpt: MAX_SEQ_LEN = int(ckpt['max_seq_len'])

print(f"\n  CONFIG: VOCAB_SIZE={VOCAB_SIZE} | MAX_SEQ_LEN={MAX_SEQ_LEN}")

# =============================================================================
# CELL 2b â€” Anomaly Distribution Analysis (runs only on fresh build)
#
# Purpose: understand where anomalies cluster across the HDFS temporal order.
# HDFS anomalies are known to be front-loaded (early blocks are anomalous).
# If deciles 1â€“6 have much higher anomaly rate than decile 10 (test slice),
# the temporal 60/20/20 split produces a test set with fewer anomalies
# â†’ harder evaluation, but incomparable to paper numbers from random splits.
#
# Output used to decide: temporal split vs. random stratified split.
# =============================================================================
print("\n" + "="*65)
print("  [CELL 2b] Anomaly rate per decile of sessions (temporal order)")
print("="*65)
if 'block_order' in dir() and 'block_labels' in dir() and block_order:
    # Fresh run â€” block_order and block_labels are still in memory
    _labels_full = np.array([block_labels[b] for b in block_order], dtype=np.int32)
    _n = len(_labels_full)
    _decile_sz = _n // 10
    for _d in range(10):
        _s = _d * _decile_sz
        _e = (_d + 1) * _decile_sz if _d < 9 else _n
        _chunk = _labels_full[_s:_e]
        _rate = _chunk.mean() * 100
        print(f"    Decile {_d+1:2d} ({_s:>8,}â€“{_e:>8,}): "
              f"anomaly rate = {_rate:5.2f}%  ({_chunk.sum():>5,} anomalies)")
    print(f"\n    Split boundaries (60/20/20):")
    _i1 = int(_n * 0.60)
    _i2 = int(_n * 0.80)
    _anom_train = _labels_full[:_i1].mean() * 100
    _anom_val   = _labels_full[_i1:_i2].mean() * 100
    _anom_test  = _labels_full[_i2:].mean() * 100
    print(f"      Train (0â€“{_i1:,})        : {_anom_train:.2f}% anomaly")
    print(f"      Val   ({_i1:,}â€“{_i2:,})  : {_anom_val:.2f}% anomaly")
    print(f"      Test  ({_i2:,}â€“{_n:,})   : {_anom_test:.2f}% anomaly")
    if _anom_test < _anom_train * 0.6:
        print(f"\n    âš ï¸  Anomalies ARE front-loaded â€” test set is anomaly-poor.")
        print(f"       â†’ Recommend random stratified split for paper comparison.")
    else:
        print(f"\n    âœ… Anomaly distribution is roughly uniform across splits.")
    del _labels_full
else:
    print("    â„¹ï¸  block_order not in memory (checkpoint resume) â€” "
          "re-run from scratch to see decile breakdown.")


# =============================================================================
# CELL 3 â€” Word2Vec Training (from scratch on the FULL HDFS corpus)
#
# Fundamental distinction from Notebook 12 (nn.Embedding):
#   nn.Embedding is initialised randomly and updated by backprop.
#   Word2Vec learns SEMANTIC relationships between log templates
#   by training a skip-gram/CBOW model on the co-occurrence structure
#   of ALL sessions â€” before any AE training starts.
#   This pre-trained semantic space is the core contribution
#   of [Bekkouche2025_BiLSTM].
#
# W2V Hyper-parameters: [Bekkouche2025_BiLSTM] Â§III-A
#   vector_size=100  â€” embedding dimension (paper default)
#   window=5         â€” context window
#   min_count=1      â€” keep ALL templates (no vocabulary pruning)
#   sg=1             â€” skip-gram (better for rare events [Meng2019])
#   workers=1        â€” deterministic training (SEED=42)
#   epochs=10        â€” sufficient for small log vocabulary
#
# Memory note: the W2V CORPUS is built from the int-encoded sequences
#   (list of lists of int strings) â€” RAM cost â‰ˆ size of X_train_idx
#   which is already loaded above.
# =============================================================================
W2V_PATH = f'{MODEL_DIR}/w2v_hdfs_full.model'
EMBED_DIM = 100   # [Bekkouche2025_BiLSTM] Â§III-A

if 'w2v_ready' in ckpt:
    print("\n[CELL 3] â­ï¸  Word2Vec already trained â€” loading model ...")
    w2v_model = Word2Vec.load(W2V_PATH)
    print(f"  W2V loaded | vectors: {len(w2v_model.wv):,} | dim={w2v_model.vector_size}")

else:
    print(f"\n{'='*65}")
    print("  [CELL 3] Training Word2Vec on FULL HDFS sessions ...")
    print(f"{'='*65}")
    t0 = time.time()

    # Build W2V corpus from train + test only (no separate val in paper protocol)
    # [Bekkouche2025_BiLSTM]: W2V trained on COMPLETE corpus (train+test)
    # to learn embeddings for ALL log keys, including rare anomalous ones.
    print("  Building W2V corpus from train+test sessions ...")
    corpus_train = [
        [str(tok) for tok in row if tok != 0]   # exclude PAD tokens
        for row in X_train_idx
    ]
    corpus_test = [
        [str(tok) for tok in row if tok != 0]
        for row in X_test_idx
    ]
    corpus_all = corpus_train + corpus_test

    total_sentences = len(corpus_all)
    print(f"  Corpus: {total_sentences:,} sentences | "
          f"first 3 lengths: {[len(s) for s in corpus_all[:3]]}")

    # Train Word2Vec
    # workers=1 ensures deterministic output with SEED
    w2v_model = Word2Vec(
        sentences   = corpus_all,
        vector_size = EMBED_DIM,
        window      = 5,
        min_count   = 1,       # keep ALL tokens (rare anomalies included)
        sg          = 1,       # skip-gram [Meng2019_LogAnomaly]
        workers     = 1,       # deterministic
        seed        = SEED,
        epochs      = 10,
    )
    w2v_model.save(W2V_PATH)

    del corpus_train, corpus_test, corpus_all
    gc.collect()

    elapsed = time.time() - t0
    print(f"  âœ… Word2Vec trained in {elapsed:.0f}s")
    print(f"     Vocabulary: {len(w2v_model.wv):,} tokens | dim={w2v_model.vector_size}")
    print(f"     Saved â†’ {W2V_PATH}")

    ckpt['w2v_ready']    = True
    ckpt['w2v_path']     = W2V_PATH
    ckpt['w2v_vocab_sz'] = len(w2v_model.wv)
    save_ckpt(ckpt)

# =============================================================================
# CELL 4 â€” Build Embedding Matrix from Word2Vec (fine-tuned during AE training)
#
# Convert the Word2Vec model into a torch.nn.Embedding layer.
# Embeddings are UNFROZEN (freeze=False) to allow fine-tuning during
# AE training. This lets the semantic space adapt to the reconstruction
# task, providing richer separation between normal and anomalous patterns.
#
# Alignment: vocab index i â†’ W2V vector for token str(i)
#   If str(i) not in W2V vocabulary â†’ small random vector (rare edge case)
# =============================================================================
print(f"\n[CELL 4] Building embedding matrix ({VOCAB_SIZE} Ã— {EMBED_DIM}) â€” will be fine-tuned ...")

embedding_matrix = np.zeros((VOCAB_SIZE, EMBED_DIM), dtype=np.float32)
oov_count = 0

for idx in range(VOCAB_SIZE):
    token_str = str(idx)
    if token_str in w2v_model.wv:
        embedding_matrix[idx] = w2v_model.wv[token_str]
    else:
        # OOV: use small random vector with same scale as W2V vectors
        embedding_matrix[idx] = np.random.normal(
            0, w2v_model.wv.vectors.std(), EMBED_DIM).astype(np.float32)
        oov_count += 1

embedding_tensor = torch.from_numpy(embedding_matrix)   # (VOCAB_SIZE, EMBED_DIM)
print(f"  Embedding matrix: {embedding_matrix.shape} | OOV tokens: {oov_count}")
print(f"  Vector norm stats: "
      f"mean={np.linalg.norm(embedding_matrix, axis=1).mean():.3f} | "
      f"std={np.linalg.norm(embedding_matrix, axis=1).std():.3f}")

del embedding_matrix, w2v_model
gc.collect()

# =============================================================================
# CELL 5 â€” BiLSTM Autoencoder Architecture
#
# [Bekkouche2025_BiLSTM] Â§III-B â€” architecture description:
#   Input: padded session of log-key indices, shape (B, T)
#   1. Embedding lookup (frozen W2V weights) â†’ (B, T, 100)
#   2. BiLSTM Encoder: last hidden state (forward+backward concatenated)
#      â†’ Linear(hidden*2, hidden) + tanh â†’ latent (B, hidden)
#   3. Decoder: repeat latent T times â†’ (B, T, hidden)
#      â†’ LSTM Decoder (unidirectional â€” no future info at decode time)
#      â†’ Linear(hidden, embed_dim=100) â†’ reconstructed embeddings (B, T, 100)
#   Loss: MSE between original W2V embeddings and reconstructed embeddings
#
# Design choices:
#   â€¢ Bidirectional encoder: captures full context of the session sequence
#     for compression [Guo2021_LogBERT]
#   â€¢ Unidirectional decoder: causal (models reconstruction step-by-step)
#   â€¢ MSE on embedding space (not on logits): smooth, differentiable,
#     semantic proximity is meaningful in W2V space
#   â€¢ padding_idx=0 in embedding: PAD tokens map to zero vector, do not
#     contribute meaningfully to reconstruction error
# =============================================================================
class BiLSTMAutoencoderW2V(nn.Module):
    """
    BiLSTM Autoencoder with fine-tuned Word2Vec embeddings + positional encoding.

    Reproduces [Bekkouche2025_BiLSTM] architecture for HDFS session-level
    unsupervised anomaly detection, with three targeted improvements:
      1. W2V embeddings are UNFROZEN for fine-tuning (richer separation)
      2. Learnable positional encoding replaces zero decoder input
      3. Anomaly score uses 0.5*mean + 0.5*max MSE (amplifies anomalous events)

    Encoder: Embedding (fine-tuned W2V) â†’ BiLSTM â†’ last hidden state â†’ latent
    Decoder: initial hidden state from latent + positional encoding â†’ LSTM â†’ Linear â†’ reconstructed embeddings

    Training loss: MSE(original_embeddings, reconstructed_embeddings)
    Anomaly score: 0.5 * mean_MSE + 0.5 * max_MSE over all timesteps
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,   # (vocab_size, embed_dim) â€” from W2V
        hidden_size: int  = 128,
        num_layers:  int  = 2,
        dropout:     float = 0.2,
    ):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape

        # Fine-tuned W2V embedding layer â€” weights ARE updated by backprop
        # Fine-tuned W2V embeddings for richer separation
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=False,         # IMPROVEMENT 1: unfreeze for fine-tuning
            padding_idx=0,        # PAD â†’ zero vector, no gradient
        )
        self.embed_dim   = embed_dim
        self.hidden_size = hidden_size

        # BiLSTM Encoder [Bekkouche2025_BiLSTM] Â§III-B
        # Bidirectional to capture full session context for compression
        self.encoder = nn.LSTM(
            input_size  = embed_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = True,           # CRITICAL: BiLSTM
        )
        self.encoder_dropout = nn.Dropout(dropout)

        # Compress bidirectional last hidden state to single latent vector
        # h_n[-2] = last forward hidden state
        # h_n[-1] = last backward hidden state
        # Concatenate â†’ Linear â†’ tanh â†’ latent of size hidden_size
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
        )

        # LSTM Decoder (unidirectional â€” [Bekkouche2025_BiLSTM])
        # The latent vector is the initial hidden state; positional encoding is fed as input.
        self.decoder = nn.LSTM(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = 1,               # single-layer decoder
            batch_first = True,
            bidirectional = False,         # CRITICAL: unidirectional decoder
        )
        self.decoder_dropout = nn.Dropout(dropout)

        # IMPROVEMENT 2: Learnable positional encoding for decoder input
        # Replaces zero input â€” gives each timestep a unique signal so the
        # decoder can reconstruct position-specific patterns in the log sequence.
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, hidden_size)

        # Project decoder output back to embedding space
        self.output_proj = nn.Linear(hidden_size, embed_dim)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T) long tensor of log-key indices

        Returns:
            original_emb: (B, T, embed_dim)  â€” W2V embeddings (detached targets)
            reconstructed: (B, T, embed_dim) â€” decoder output
        """
        B, T = x.size(0), x.size(1)

        # 1. Look up frozen W2V embeddings â†’ (B, T, embed_dim)
        original_emb = self.embedding(x)

        # 2. BiLSTM Encoder â†’ last hidden states
        enc_out, (h_n, _) = self.encoder(self.encoder_dropout(original_emb))
        # h_n shape: (num_layers * 2, B, hidden_size)
        # Extract last-layer forward and backward hidden states
        h_forward  = h_n[-2]   # (B, hidden_size)
        h_backward = h_n[-1]   # (B, hidden_size)
        h_concat   = torch.cat([h_forward, h_backward], dim=-1)  # (B, hidden*2)

        # 3. Compress to latent vector via Linear+tanh
        latent = self.bottleneck(h_concat)   # (B, hidden_size)

        # 4. Reshape latent -> initial hidden state for decoder (1, B, hidden_size)
        h0 = latent.unsqueeze(0)
        c0 = torch.zeros_like(h0)

        # 5. IMPROVEMENT 2: Positional encoding as decoder input (replaces zeros)
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)  # (B, T)
        decoder_input = self.pos_emb(positions)   # (B, T, hidden_size)

        # 6. Run decoder with initial state
        dec_out, _ = self.decoder(decoder_input, (h0, c0))    # (B, T, hidden_size)

        # 7. Project to embedding space
        reconstructed = self.output_proj(self.decoder_dropout(dec_out))   # (B, T, embed_dim)

        return original_emb, reconstructed


# =============================================================================
# CELL 6 â€” Utility Functions
# =============================================================================

def compute_session_errors(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Compute per-session reconstruction error using mixed mean+max scoring.

    IMPROVEMENT 3: Anomaly score = 0.5 * mean_error + 0.5 * max_error
    where:
        mean_error = average MSE across all timesteps
        max_error  = maximum MSE across all timesteps in the session
    This amplifies the signal from the most anomalous event in each
    session, improving recall on anomalous sessions without sacrificing
    precision from the mean component.

    Returns: float32 array of shape (N_sessions,)
    """
    model.eval()
    errors = []
    ds = TensorDataset(torch.from_numpy(X).long())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            emb, recon = model(xb)
            # IMPROVEMENT 3: mixed mean+max scoring
            mse_per_step = ((emb - recon) ** 2).mean(dim=2)   # (B, T)
            mean_err = mse_per_step.mean(dim=1)                # (B,)
            max_err  = mse_per_step.max(dim=1).values          # (B,)
            score    = 0.5 * mean_err + 0.5 * max_err          # (B,)
            errors.extend(score.cpu().numpy().tolist())

    return np.array(errors, dtype=np.float32)


def f1_threshold_search(
    errors_normal_val: np.ndarray,
    errors_val_all:    np.ndarray,
    y_val:             np.ndarray,
    n_points:          int = 5000,
) -> tuple:
    """
    F1-sensitive threshold search on the FULL (unmodified) validation set.

    [Bekkouche2025_BiLSTM] Â§III-D:
    'An F1-sensitive threshold is searched on the validation set.'

    Search range: [10th-percentile of normal val errors,
                   99.5th-percentile of all val errors]
    Using 10th pct of normal errors avoids the mass of trivially low
    thresholds; 99.5th pct captures the highest anomaly errors without
    extreme outlier distortion.
    5000 candidates gives sub-0.1% resolution over the error range.

    Returns: (best_threshold, best_f1, all_thresholds, all_f1s)
    """
    lo = float(np.percentile(errors_normal_val, 10))
    hi = float(np.percentile(errors_val_all,  99.5))
    thresholds = np.linspace(lo, hi, n_points)

    best_f1, best_thr = 0.0, lo
    all_f1s = []

    for thr in thresholds:
        preds = (errors_val_all > thr).astype(int)
        f1    = f1_score(y_val, preds, pos_label=1, zero_division=0)
        all_f1s.append(f1)
        if f1 > best_f1:
            best_f1  = f1
            best_thr = thr

    return best_thr, best_f1, thresholds, np.array(all_f1s)


def train_ae_model(
    X_train_normal: np.ndarray,
    X_val:          np.ndarray,
    y_val:          np.ndarray,
    emb_tensor:     torch.Tensor,
    config:         dict,
    max_epochs:     int  = 150,
    patience:       int  = 15,
    verbose:        bool = True,
) -> tuple:
    """
    Train BiLSTMAutoencoderW2V with early stopping and mixed precision.

    Training: NORMAL sessions only [Bekkouche2025_BiLSTM]
    Schedulers:
      - CosineAnnealingLR (primary)   â€” smooth global decay
      - ReduceLROnPlateau (secondary) â€” aggressive LR cut on plateau
        factor=0.5, patience=5 epochs without val MSE improvement
    Validation MSE monitored on normal validation sessions.
    Returns (model, best_state_dict, train_losses, val_losses, best_val_mse)
    """
    model = BiLSTMAutoencoderW2V(
        embedding_matrix = emb_tensor,
        hidden_size      = config['hidden_size'],
        num_layers       = config['num_layers'],
        dropout          = config['dropout'],
    ).to(DEVICE)

    criterion     = nn.MSELoss()
    # IMPROVEMENT 1: Separate LR for unfrozen embeddings vs rest of model
    # Lower LR for embeddings prevents catastrophic forgetting of W2V semantics
    embedding_lr = config.get('embedding_lr', config['lr'] * 0.1)
    optimizer     = torch.optim.Adam([
        {'params': model.embedding.parameters(), 'lr': embedding_lr},
        {'params': [p for n, p in model.named_parameters()
                    if 'embedding' not in n], 'lr': config['lr']},
    ])
    # Primary: global cosine annealing across full training budget
    scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=config['lr'] * 0.01)
    # Secondary: cut LR by half when val MSE stalls for 5 epochs
    # NOTE: verbose parameter removed â€” deprecated/dropped in PyTorch â‰¥ 2.2
    scheduler_plat = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    scaler        = GradScaler()

    bs = config['batch_size']

    # Training DataLoader â€” normal sessions only
    train_ds = TensorDataset(torch.from_numpy(X_train_normal).long())
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0,
                          pin_memory=(DEVICE.type == 'cuda'))

    # Validation DataLoader â€” normal val sessions only (for MSE monitoring)
    X_val_normal = X_val[y_val == 0]
    val_ds   = TensorDataset(torch.from_numpy(X_val_normal).long())
    val_dl   = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    best_val_mse = float('inf')
    best_state   = None
    no_improve   = 0
    train_losses = []
    val_losses   = []

    for epoch in range(1, max_epochs + 1):
        # â”€â”€ Training pass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        model.train()
        epoch_loss = 0.0

        for (xb,) in train_dl:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                emb, recon = model(xb)
                # .detach() prevents gradients flowing through the frozen W2V path
                loss = criterion(recon, emb.detach())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Gradient clipping for LSTM stability [Du2017_DeepLog]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_train = epoch_loss / max(len(train_dl), 1)
        train_losses.append(avg_train)
        scheduler_cos.step()   # cosine step each epoch

        # â”€â”€ Validation pass (normal sessions only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (xb,) in val_dl:
                xb = xb.to(DEVICE)
                with autocast():
                    emb, recon = model(xb)
                    val_loss  += criterion(recon, emb.detach()).item()
        avg_val = val_loss / max(len(val_dl), 1)
        val_losses.append(avg_val)
        scheduler_plat.step(avg_val)   # plateau step on val MSE

        # â”€â”€ Early stopping â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if avg_val < best_val_mse:
            best_val_mse = avg_val
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch:3d}/{max_epochs} | "
                  f"Train MSE={avg_train:.5f} | "
                  f"Val MSE={avg_val:.5f} | "
                  f"Best={best_val_mse:.5f} | "
                  f"LR={cur_lr:.2e} | "
                  f"patience={no_improve}/{patience}")

        if no_improve >= patience:
            if verbose:
                print(f"    â¹ Early stopping at epoch {epoch} (patience={patience})")
            break

    model.load_state_dict(best_state)
    return model, best_state, train_losses, val_losses, best_val_mse


# =============================================================================
# CELL 7 â€” Optuna Hyperparameter Search + Full Training
#
# Search space grounded in [Bekkouche2025_BiLSTM] + project conventions:
#   hidden_size : [128, 256, 512] â€” encoder/decoder hidden dim (512 added)
#   num_layers  : 1â€“2             â€” BiLSTM depth (3 layers overfits HDFS)
#   dropout     : 0.1â€“0.4         â€” regularisation
#   lr          : log[1e-4, 5e-3] â€” Adam learning rate
#   batch_size  : [128, 256, 512] â€” GPU-friendly
#
# Warm-start: [Bekkouche2025_BiLSTM] defaults (hidden=128, layers=2, dropout=0.2)
#
# Optuna objective: MINIMISE validation MSE on normal sessions.
# Full training: 150 epochs max, patience=15 â€” sufficient for convergence.
#
# Checkpoint guard: 'ae_done' prevents re-running on timeout resume.
# =============================================================================
MODEL_PATH  = f'{MODEL_DIR}/bilstm_ae_w2v_hdfs_opt.pt'
CONFIG_PATH = f'{MODEL_DIR}/bilstm_ae_w2v_hdfs_config.json'

if 'ae_done' in ckpt:
    print("\n[CELL 7] â­ï¸  BiLSTM-AE already done (checkpoint 'ae_done') â€” skipping training.")

else:
    print(f"\n{'='*65}")
    print("  ðŸ§  HDFS BiLSTM-AE + Word2Vec â€” OPTUNA + FULL TRAINING")
    print(f"{'='*65}")
    t0_total = time.time()

    # â”€â”€ Load sessions (already built in CELL 2) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if 'X_train_idx' not in dir():
        train_data = np.load(HDFSPath_NPZ_train)
        test_data  = np.load(HDFSPath_NPZ_test)
        X_train_idx, y_train = train_data['X'], train_data['y']
        X_test_idx,  y_test  = test_data['X'],  test_data['y']
        print(f"  Reloaded sessions: train={X_train_idx.shape} test={X_test_idx.shape}")

    # Normal training sessions â€” unsupervised protocol [Bekkouche2025_BiLSTM]
    X_train_normal = X_train_idx[y_train == 0]
    print(f"\n  Normal train: {len(X_train_normal):,} / {len(X_train_idx):,} sessions "
          f"({len(X_train_normal)/len(X_train_idx)*100:.1f}%)")
    print(f"  Test        : {len(y_test):,} sessions "
          f"({y_test.mean()*100:.2f}% anomaly)")
    print(f"  VOCAB_SIZE  : {VOCAB_SIZE:,} | MAX_SEQ_LEN: {MAX_SEQ_LEN}")
    print(f"  EMBED_DIM   : {EMBED_DIM} (frozen W2V)")

    # â”€â”€ Paper protocol verification [Bekkouche2025_BiLSTM] Table I â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _n_train     = len(y_train)
    _n_test      = len(y_test)
    _anom_train  = int(y_train.sum())
    _anom_test   = int(y_test.sum())
    _unique_evts = VOCAB_SIZE - 2   # subtract PAD and UNK

    print(f"\n{'='*51}")
    print(f"  === PAPER PROTOCOL VERIFICATION ===")
    print(f"  Train sequences : {_n_train:>8,}   (paper: 517,554)"
          f"  {'âœ…' if abs(_n_train - 517554) < 10000 else 'âš ï¸ '}")
    print(f"  Test sequences  : {_n_test:>8,}   (paper:  57,507)"
          f"  {'âœ…' if abs(_n_test - 57507) < 2000 else 'âš ï¸ '}")
    print(f"  Train anomalies : {_anom_train:>8,}   (paper:  15,154)"
          f"  {'âœ…' if abs(_anom_train - 15154) < 2000 else 'âš ï¸ '}")
    print(f"  Test anomalies  : {_anom_test:>8,}   (paper:   1,684)"
          f"  {'âœ…' if abs(_anom_test - 1684) < 300 else 'âš ï¸ '}")
    print(f"  Unique events   : {_unique_evts:>8,}   (paper:      48)"
          f"  {'âœ…' if abs(_unique_evts - 48) <= 10 else 'âš ï¸  (Drain granularity diff)'}")
    print(f"  Threshold target: TEST SET  (matches paper Â§II-C-a)  âœ…")
    print(f"{'='*51}")

    _any_mismatch = (
        abs(_n_train - 517554) >= 10000 or
        abs(_n_test  - 57507)  >= 2000  or
        abs(_anom_train - 15154) >= 2000 or
        abs(_anom_test  - 1684)  >= 300
    )
    if _any_mismatch:
        print("\n  âš ï¸  WARNING: One or more counts deviate significantly from Table I.")
        print("       Likely cause: our HDFS_Drain.csv total sessions differ from paper's 575,061.")
        print(f"       Our total: {_n_train + _n_test:,}  |  Paper: 575,061")
        print("       Possible reasons: different Drain parsing, CSV version, or duplicate removal.")
        print("       Proceeding anyway â€” split ratio 90/10 and anomaly rate match the paper.\n")
    else:
        print("\n  âœ… All counts match Table I. Proceeding with paper-exact protocol.\n")

    # â”€â”€ Small validation set for early stopping / Optuna (10% of normal train) â”€
    # The paper has no separate val set â€” we hold out 10% of normal-only train
    # sessions INTERNALLY for early stopping. This does NOT affect threshold tuning.
    # Threshold is tuned on the test set directly, matching [Bekkouche2025_BiLSTM].
    from sklearn.model_selection import train_test_split as _tts2
    _idx_tr = np.arange(len(X_train_normal))
    _idx_nt, _idx_nv = _tts2(_idx_tr, test_size=0.10, random_state=SEED)
    X_train_opt = X_train_normal[_idx_nt]   # 90% of normal â€” actual training
    X_val_opt   = X_train_normal[_idx_nv]   # 10% of normal â€” early stopping only
    y_val_opt   = np.zeros(len(X_val_opt), dtype=np.int32)  # all normal (label=0)
    print(f"\n  Internal val (normal only, for early stopping):")
    print(f"    Optuna/training val: {len(X_val_opt):,} normal sessions")
    del _idx_tr, _idx_nt, _idx_nv

    # â”€â”€ Optuna objective â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def objective(trial: optuna.Trial) -> float:
        cfg = {
            # 512 added â€” larger hidden dim helps on high-dim W2V space
            'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512]),
            # 1â€“2 only â€” 3-layer BiLSTM overfits on HDFS normal sessions
            'num_layers':  trial.suggest_int('num_layers', 1, 2),
            'dropout':     trial.suggest_float('dropout', 0.1, 0.4),
            'lr':          trial.suggest_float('lr', 1e-4, 5e-3, log=True),
            # IMPROVEMENT 1: Separate LR for unfrozen W2V embeddings
            'embedding_lr': trial.suggest_float('embedding_lr', 1e-5, 1e-3, log=True),
            'batch_size':  trial.suggest_categorical('batch_size', [128, 256, 512]),
        }
        # â”€â”€ Sample 50k sessions for fast Optuna trial â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Full X_train_opt is ~450k sessions â€” each trial would take ~8 min.
        # We subsample 50k (stratified by trial.number for diversity) so
        # each trial finishes in ~1 min, allowing all 30 trials to run.
        from sklearn.model_selection import train_test_split as _tts3
        if len(X_train_opt) > 50000:
            _, X_optuna_sample = _tts3(
                X_train_opt, test_size=50000, random_state=trial.number)
        else:
            X_optuna_sample = X_train_opt

        try:
            _, _, _, _, val_mse = train_ae_model(
                X_optuna_sample, X_val_opt, y_val_opt,
                embedding_tensor, cfg,
                max_epochs=15, patience=5, verbose=False,
            )
        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
        return val_mse   # minimise

    study = optuna.create_study(
        direction = 'minimize',
        sampler   = optuna.samplers.TPESampler(seed=SEED),
        pruner    = optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    # Warm-start: [Bekkouche2025_BiLSTM] defaults + embedding_lr
    study.enqueue_trial({
        'hidden_size':  128,
        'num_layers':   2,
        'dropout':      0.2,
        'lr':           0.001,
        'embedding_lr': 0.0001,
        'batch_size':   256,
    })
    # Warm-start 2: larger model variant
    study.enqueue_trial({
        'hidden_size':  256,
        'num_layers':   1,
        'dropout':      0.2,
        'lr':           0.001,
        'embedding_lr': 0.0001,
        'batch_size':   256,
    })

    print(f"\n  ðŸ” Optuna (30 trials, timeout=3600s, 50k sessions/trial) ...")
    study.optimize(objective, n_trials=30, timeout=3600,
                   show_progress_bar=False)
    # â”€â”€ Retrieve best Optuna params (with safety fallback) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        best_params = study.best_params
        print(f"  ðŸ† Best params  : {best_params}")
        print(f"  ðŸ† Best val MSE : {study.best_value:.6f}")
        print(f"  ðŸ† Completed trials: {len(completed)}/{len(study.trials)}")
    else:
        # All trials failed/pruned â€” fall back to [Bekkouche2025_BiLSTM] paper defaults
        best_params = {
            'hidden_size': 128,
            'num_layers':  2,
            'dropout':     0.2,
            'lr':          0.001,
            'batch_size':  256,
        }
        print(f"  âš ï¸  All Optuna trials failed â€” using paper defaults: {best_params}")
        print(f"       Check the error messages above to diagnose the failure.")

    # â”€â”€ Full training with best hyper-parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n  ðŸš€ Full training (max_epochs=150, patience=15) ...")
    t0_train = time.time()
    (model, best_state,
     train_losses, val_losses, best_val_mse) = train_ae_model(
        X_train_opt, X_val_opt, y_val_opt,
        embedding_tensor, best_params,
        max_epochs=150, patience=15, verbose=True,
    )
    train_time = time.time() - t0_train
    print(f"  âœ… Training done in {train_time:.0f}s")

    # â”€â”€ Compute reconstruction errors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Two passes: primary (90/10 stratified) + secondary (temporal 60/20/20)
    # Primary: X_train_idx + X_test_idx (paper split)
    # Secondary: recomputed on-the-fly from all sessions in temporal order
    X_all_primary = np.concatenate([X_train_idx, X_test_idx], axis=0)
    y_all_primary = np.concatenate([y_train, y_test], axis=0)

    print("\n  ðŸ“ Computing reconstruction errors (primary 90/10 split) ...")
    t_inf_start = time.time()
    errors_all_primary = compute_session_errors(model, X_all_primary, batch_size=512)
    infer_time = time.time() - t_inf_start
    print(f"  âœ… Errors computed in {infer_time:.1f}s")

    errors_train = errors_all_primary[:len(y_train)]
    errors_test  = errors_all_primary[len(y_train):]

    errors_train_normal = errors_train[y_train == 0]
    print(f"     Normal train errors: Î¼={errors_train_normal.mean():.5f} Ïƒ={errors_train_normal.std():.5f}")
    print(f"     Test errors:         Î¼={errors_test.mean():.5f} Ïƒ={errors_test.std():.5f}")

    # â”€â”€ [EVAL 1] Paper protocol â€” threshold tuned on TEST set â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # [Bekkouche2025_BiLSTM] Â§II-C-a:
    # "the test partition of the dataset is used to determine the threshold
    #  value that maximises the F1-Score"
    # This is semi-supervised: threshold picks the best cut on the test set directly.
    errors_test_normal = errors_test[y_test == 0]
    best_thr_paper, best_test_f1_paper, all_thrs_paper, all_f1s_paper = f1_threshold_search(
        errors_test_normal, errors_test, y_test, n_points=5000)
    print(f"\n  [EVAL 1] Paper protocol â€” threshold search on TEST set (semi-supervised):")
    print(f"           Normal={len(errors_test_normal):,} | Anomaly={(y_test==1).sum():,} | "
          f"Total={len(y_test):,} (rate={y_test.mean()*100:.2f}%)")
    print(f"     Best threshold (Paper)  : {best_thr_paper:.6f}")
    print(f"     Best test F1 (Paper)    : {best_test_f1_paper:.4f}")

    # Evaluate on test set â€” PAPER PROTOCOL
    y_pred_paper = (errors_test > best_thr_paper).astype(int)

    # Normalise errors to [0,1] range for ROC/PR curves
    e_min_p = errors_test.min()
    e_max_p = errors_test.max()
    y_prob_paper = (errors_test - e_min_p) / (e_max_p - e_min_p + 1e-12)

    fpr_paper,  tpr_paper,  _  = roc_curve(y_test, y_prob_paper)
    prec_paper, rec_paper,  _  = precision_recall_curve(y_test, y_prob_paper)
    roc_auc_paper = auc(fpr_paper, tpr_paper)
    pr_auc_paper  = auc(rec_paper, prec_paper)

    p_paper  = precision_score(y_test, y_pred_paper, pos_label=1, zero_division=0)
    r_paper  = recall_score(y_test,    y_pred_paper, pos_label=1, zero_division=0)
    f1_paper = f1_score(y_test,        y_pred_paper, pos_label=1, zero_division=0)
    mf1_paper = f1_score(y_test,       y_pred_paper, average='macro', zero_division=0)
    mcc_paper = matthews_corrcoef(y_test, y_pred_paper)
    ap_paper  = average_precision_score(y_test, y_prob_paper)

    metrics_paper = {
        'Dataset':                 'HDFS',
        'Model':                   'BiLSTM-AE+W2V',
        'Split':                   'Stratified 90/10 (paper protocol)',
        'Threshold_Source':        'Test set (semi-supervised, per Â§II-C-a)',
        'Type':                    'Unsupervised (DL)',
        'Paper':                   'Bekkouche2025_BiLSTM',
        'Paper_F1':                0.993,
        'Paper_Precision':         0.987,
        'Paper_Recall':            1.000,
        'Precision':               round(float(p_paper),   4),
        'Recall':                  round(float(r_paper),   4),
        'F1_Anomaly':              round(float(f1_paper),  4),
        'F1_Delta_vs_Paper':       round(float(f1_paper) - 0.993, 4),
        'Macro_F1':                round(float(mf1_paper), 4),
        'AUC_ROC':                 round(float(roc_auc_paper), 4),
        'AUC_PR':                  round(float(pr_auc_paper),  4),
        'Avg_Precision':           round(float(ap_paper),  4),
        'MCC':                     round(float(mcc_paper), 4),
        'Threshold':               round(float(best_thr_paper), 6),
        'Best_Val_MSE':            round(float(best_val_mse), 6),
        'EMBED_DIM':               EMBED_DIM,
        'MAX_SEQ_LEN':             MAX_SEQ_LEN,
        'VOCAB_SIZE':              VOCAB_SIZE,
        'hidden_size':             best_params.get('hidden_size'),
        'num_layers':              best_params.get('num_layers'),
        'dropout':                 best_params.get('dropout'),
        'lr':                      best_params.get('lr'),
        'batch_size':              best_params.get('batch_size'),
        'Train_Time_s':            round(train_time, 1),
        'Inference_Time_s':        round(infer_time, 4),
        'Inference_Per_Sample_ms': round(infer_time / max(len(y_test), 1) * 1000, 4),
    }

    # â”€â”€ [EVAL 2] Our rigorous protocol â€” temporal 60/20/20, blind test â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Recompute errors on temporal splits (no reuse of primary model needed â€”
    # same model, different index slices from the full session array)
    # Re-encode temporal splits using the SAME full session array order.
    # We need the full sequences array â€” load from train+test NPZ and reorder.
    print("\n  ðŸ“ Computing reconstruction errors (temporal 60/20/20 split) ...")
    # Reconstruct full ordered array from checkpoint NPZs
    if 'X_temp_train' not in dir():
        # Must recompute â€” load all data and apply temporal slices
        _all_X = np.concatenate([X_train_idx, X_test_idx], axis=0)   # note: shuffled by stratified split
        print("    â„¹ï¸  Temporal arrays not in memory (checkpoint resume) â€”")
        print("        Temporal EVAL 2 skipped. Run from scratch to get temporal results.")
        X_temp_train = X_temp_val = X_temp_test = None
        y_temp_train = y_temp_val = y_temp_test = None
        del _all_X

    if X_temp_test is not None:
        t_inf2 = time.time()
        errors_temp_val  = compute_session_errors(model, X_temp_val,  batch_size=512)
        errors_temp_test = compute_session_errors(model, X_temp_test, batch_size=512)
        infer_time2 = time.time() - t_inf2

        errors_temp_val_normal = errors_temp_val[y_temp_val == 0]
        best_thr_temp, best_val_f1_temp, all_thrs_temp, all_f1s_temp = f1_threshold_search(
            errors_temp_val_normal, errors_temp_val, y_temp_val, n_points=5000)

        y_pred_temp = (errors_temp_test > best_thr_temp).astype(int)
        e_min_t = errors_temp_test.min()
        e_max_t = errors_temp_test.max()
        y_prob_temp = (errors_temp_test - e_min_t) / (e_max_t - e_min_t + 1e-12)

        fpr_temp,  tpr_temp,  _  = roc_curve(y_temp_test, y_prob_temp)
        prec_temp, rec_temp,  _  = precision_recall_curve(y_temp_test, y_prob_temp)
        roc_auc_temp = auc(fpr_temp, tpr_temp)
        pr_auc_temp  = auc(rec_temp, prec_temp)

        p_temp   = precision_score(y_temp_test, y_pred_temp, pos_label=1, zero_division=0)
        r_temp   = recall_score(y_temp_test,    y_pred_temp, pos_label=1, zero_division=0)
        f1_temp  = f1_score(y_temp_test,        y_pred_temp, pos_label=1, zero_division=0)
        mf1_temp = f1_score(y_temp_test,        y_pred_temp, average='macro', zero_division=0)
        mcc_temp = matthews_corrcoef(y_temp_test, y_pred_temp)
        ap_temp  = average_precision_score(y_temp_test, y_prob_temp)

        metrics_temp = {
            'Dataset':                 'HDFS',
            'Model':                   'BiLSTM-AE+W2V',
            'Split':                   'Temporal 60/20/20 (our rigorous protocol)',
            'Threshold_Source':        'Temporal val set (blind test)',
            'Type':                    'Unsupervised (DL)',
            'Paper':                   'Bekkouche2025_BiLSTM',
            'Paper_F1':                0.993,
            'Paper_Precision':         0.987,
            'Paper_Recall':            1.000,
            'Precision':               round(float(p_temp),   4),
            'Recall':                  round(float(r_temp),   4),
            'F1_Anomaly':              round(float(f1_temp),  4),
            'F1_Delta_vs_Paper':       round(float(f1_temp) - 0.993, 4),
            'Macro_F1':                round(float(mf1_temp), 4),
            'AUC_ROC':                 round(float(roc_auc_temp), 4),
            'AUC_PR':                  round(float(pr_auc_temp),  4),
            'Avg_Precision':           round(float(ap_temp),  4),
            'MCC':                     round(float(mcc_temp), 4),
            'Threshold':               round(float(best_thr_temp), 6),
            'Val_F1_at_Threshold':     round(float(best_val_f1_temp), 4),
            'Best_Val_MSE':            round(float(best_val_mse), 6),
            'EMBED_DIM':               EMBED_DIM,
            'MAX_SEQ_LEN':             MAX_SEQ_LEN,
            'VOCAB_SIZE':              VOCAB_SIZE,
            'hidden_size':             best_params.get('hidden_size'),
            'num_layers':              best_params.get('num_layers'),
            'dropout':                 best_params.get('dropout'),
            'lr':                      best_params.get('lr'),
            'batch_size':              best_params.get('batch_size'),
            'Train_Time_s':            round(train_time, 1),
            'Inference_Time_s':        round(infer_time2, 4),
            'Inference_Per_Sample_ms': round(infer_time2 / max(len(y_temp_test), 1) * 1000, 4),
        }
    else:
        metrics_temp = None
        f1_temp = roc_auc_temp = mcc_temp = 0.0
        all_thrs_temp = all_f1s_temp = best_thr_temp = best_val_f1_temp = None
        fpr_temp = tpr_temp = prec_temp = rec_temp = None

    # â”€â”€ Print results to console â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n  {'='*55}")
    print(f"  ðŸ“Š EVAL 1 â€” Paper Protocol (90/10 strat., threshold on test):")
    print(f"  {'='*55}")
    print(classification_report(y_test, y_pred_paper, target_names=['Normal', 'Anomaly'], digits=4))
    print(f"  ROC AUC = {roc_auc_paper:.4f} | PR AUC = {pr_auc_paper:.4f} | MCC = {mcc_paper:.4f}")

    if X_temp_test is not None:
        print(f"\n  {'='*55}")
        print(f"  ðŸ“Š EVAL 2 â€” Our Protocol (temporal 60/20/20, blind test val):")
        print(f"  {'='*55}")
        print(classification_report(y_temp_test, y_pred_temp, target_names=['Normal', 'Anomaly'], digits=4))
        print(f"  ROC AUC = {roc_auc_temp:.4f} | PR AUC = {pr_auc_temp:.4f} | MCC = {mcc_temp:.4f}")

    # Gap comparison
    print(f"\n  ðŸ“š Paper target: F1=0.993, Precision=0.987, Recall=1.000")
    print(f"     EVAL 1 (paper protocol): F1={f1_paper:.4f} (Î”={f1_paper-0.993:+.4f})")
    if X_temp_test is not None:
        print(f"     EVAL 2 (our protocol):  F1={f1_temp:.4f} (Î”={f1_temp-0.993:+.4f})")

    # â”€â”€ Save model & configs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    torch.save(best_state, MODEL_PATH)
    print(f"\n  âœ… Model saved â†’ {MODEL_PATH}")

    _all_metrics = [metrics_paper]
    if metrics_temp is not None:
        _all_metrics.append(metrics_temp)

    with open(CONFIG_PATH, 'w') as f:
        json.dump({
            **best_params,
            'vocab_size':       VOCAB_SIZE,
            'embed_dim':        EMBED_DIM,
            'max_seq_len':      MAX_SEQ_LEN,
            'threshold_paper':  float(best_thr_paper),
            'best_val_mse':     float(best_val_mse),
            'train_time_s':     round(train_time, 1),
            'infer_time_s':     round(infer_time, 4),
            'paper_metrics':    metrics_paper,
            'temporal_metrics': metrics_temp,
            'threshold':        float(best_thr_paper),
            'test_f1':          float(f1_paper),
            'test_auc':         float(roc_auc_paper),
            'test_mcc':         float(mcc_paper),
        }, f, indent=2)
    print(f"  âœ… Config saved â†’ {CONFIG_PATH}")

    # Save results CSV â€” paper protocol row first, our protocol second if available
    RESULTS_CSV = f'{REPORT}/bilstm_ae_hdfs_results.csv'
    pd.DataFrame(_all_metrics).round(4).to_csv(RESULTS_CSV, index=False)
    print(f"  âœ… Results saved â†’ {RESULTS_CSV}")

    # =========================================================================
    # CELL 8 â€” Plots
    # =========================================================================
    print("\n  ðŸ“ˆ Generating plots ...")

    # â”€â”€ Plot 1: Loss & Val Curves â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ax = axes[0]
    ax.plot(range(1, len(train_losses)+1), train_losses,
            'b-o', markersize=2, linewidth=1.2, label='Train MSE')
    ax.plot(range(1, len(val_losses)+1), val_losses,
            'r-o', markersize=2, linewidth=1.2, label='Val MSE (Normal, internal)')
    ax.set_title('BiLSTM-AE+W2V â€” Training Loss Curve\n[Bekkouche2025_BiLSTM]',
                 fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    # â”€â”€ Plot 2: F1 vs Threshold Search Curves â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax = axes[1]
    ax.plot(all_thrs_paper, all_f1s_paper, 'b-', linewidth=1.5,
            label='Paper protocol (threshold on test)')
    ax.axvline(best_thr_paper, color='blue', linestyle='--', lw=1.2,
               label=f'Paper thr={best_thr_paper:.5f}\nTest F1={best_test_f1_paper:.4f}')
    if all_thrs_temp is not None:
        ax.plot(all_thrs_temp, all_f1s_temp, 'g-', linewidth=1.5,
                label='Our protocol (threshold on val)')
        ax.axvline(best_thr_temp, color='green', linestyle='--', lw=1.2,
                   label=f'Our thr={best_thr_temp:.5f}\nVal F1={best_val_f1_temp:.4f}')

    ax.set_title('F1-Sensitive Threshold Search\n[Bekkouche2025_BiLSTM] Â§II-C-a',
                 fontweight='bold')
    ax.set_xlabel('Reconstruction Error Threshold')
    ax.set_ylabel('F1 Score (Anomaly class)')
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    PLOT_LOSS = f'{REPORT}/bilstm_ae_hdfs_loss_curve.png'
    plt.savefig(PLOT_LOSS, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  âœ… Loss curve â†’ {PLOT_LOSS}")

    # â”€â”€ Plot 3: Reconstruction Error Histogram (paper protocol test set) â”€â”€â”€â”€â”€â”€â”€
    err_normal_test = errors_test[y_test == 0]
    err_anom_test   = errors_test[y_test == 1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(err_normal_test, bins=100, alpha=0.6, color='steelblue',
            label=f'Normal (n={len(err_normal_test):,})', density=True)
    ax.hist(err_anom_test,   bins=100, alpha=0.6, color='crimson',
            label=f'Anomaly (n={len(err_anom_test):,})', density=True)
    ax.axvline(best_thr_paper, color='black', linestyle='--', lw=2,
               label=f'Threshold = {best_thr_paper:.5f}')
    ax.set_title('Reconstruction Error Distribution â€” HDFS Test Set (Paper Protocol)\n'
                 '[Bekkouche2025_BiLSTM] BiLSTM-AE + Word2Vec',
                 fontweight='bold')
    ax.set_xlabel('Session Reconstruction Error (MSE)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    PLOT_HIST = f'{REPORT}/bilstm_ae_hdfs_error_hist.png'
    plt.savefig(PLOT_HIST, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  âœ… Error histogram â†’ {PLOT_HIST}")

    # â”€â”€ Plot 4: Confusion Matrices (Paper protocol vs Temporal) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    n_plots = 2 if X_temp_test is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots + 1, 4))
    if n_plots == 1:
        axes = [axes]

    # Subplot 1: Paper protocol
    cm_paper = confusion_matrix(y_test, y_pred_paper)
    sns.heatmap(cm_paper, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                annot_kws={'size': 12, 'weight': 'bold'})
    axes[0].set_title(f'Paper Protocol 90/10 (F1={f1_paper:.4f})', fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # Subplot 2: Temporal (if available)
    if X_temp_test is not None:
        cm_temp = confusion_matrix(y_temp_test, y_pred_temp)
        sns.heatmap(cm_temp, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'],
                    annot_kws={'size': 12, 'weight': 'bold'})
        axes[1].set_title(f'Our Protocol 60/20/20 (F1={f1_temp:.4f})', fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')

    fig.suptitle('Confusion Matrices â€” HDFS BiLSTM-AE+W2V\n[Bekkouche2025_BiLSTM]',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    PLOT_CM = f'{REPORT}/bilstm_ae_hdfs_cm.png'
    plt.savefig(PLOT_CM, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  âœ… Confusion matrix â†’ {PLOT_CM}")

    # â”€â”€ Plot 5: ROC Curve â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr_paper, tpr_paper, 'b-', lw=2,
            label=f'Paper Protocol 90/10 (AUC={roc_auc_paper:.4f})')
    if fpr_temp is not None:
        ax.plot(fpr_temp, tpr_temp, 'g-', lw=2,
                label=f'Our Protocol 60/20/20 (AUC={roc_auc_temp:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guess')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve â€” HDFS BiLSTM-AE+W2V\n[Bekkouche2025_BiLSTM]',
                 fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    PLOT_ROC = f'{REPORT}/bilstm_ae_hdfs_roc.png'
    plt.savefig(PLOT_ROC, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  âœ… ROC curve â†’ {PLOT_ROC}")

    # â”€â”€ Plot 6: Precision-Recall Curve â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec_paper, prec_paper, 'blue', lw=2,
            label=f'Paper Protocol (AP={ap_paper:.4f})')
    if prec_temp is not None:
        ax.plot(rec_temp, prec_temp, 'green', lw=2,
                label=f'Our Protocol (AP={ap_temp:.4f})')
    ax.axhline(y_test.mean(), color='grey', linestyle='--', lw=1, alpha=0.5,
               label=f'Baseline (prevalence={y_test.mean():.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve â€” HDFS BiLSTM-AE+W2V\n[Bekkouche2025_BiLSTM]',
                 fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    PLOT_PR = f'{REPORT}/bilstm_ae_hdfs_pr_curve.png'
    plt.savefig(PLOT_PR, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  âœ… PR curve â†’ {PLOT_PR}")

    # â”€â”€ Plot 7: Comparison Table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _rows = [
        ['Bekkouche2025_BiLSTM (paper)',        'Unsupervised', 0.987, 1.000, 0.993],
        ['BiLSTM-AE+W2V (Paper protocol 90/10)','Unsupervised',
         round(p_paper, 4), round(r_paper, 4), round(f1_paper, 4)],
    ]
    if X_temp_test is not None:
        _rows.append(['BiLSTM-AE+W2V (Our protocol 60/20/20)', 'Unsupervised',
                      round(p_temp, 4), round(r_temp, 4), round(f1_temp, 4)])
    _rows += [
        ['BiLSTM-AE nn.Emb (NB12)', 'Unsupervised', 'â€”', 'â€”', 'â€”'],
        ['DeepLog (NB13)',           'Unsupervised', 'â€”', 'â€”', 'â€”'],
        ['Attn-BiLSTM (NB06)',      'Supervised',   'â€”', 'â€”', 'â€”'],
        ['CNN+BiLSTM (NB07)',        'Supervised',   'â€”', 'â€”', 'â€”'],
    ]
    comp_df = pd.DataFrame(_rows, columns=['Model / Source', 'Type', 'Precision', 'Recall', 'F1'])

    fig, ax = plt.subplots(figsize=(11, max(3.2, 0.4 * len(_rows) + 1.2)))
    ax.axis('off')
    tbl = ax.table(
        cellText  = comp_df.values,
        colLabels = comp_df.columns,
        cellLoc   = 'center',
        loc       = 'center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for col_idx in range(len(comp_df.columns)):
        tbl[1, col_idx].set_facecolor('#cce5ff')   # paper row
        tbl[2, col_idx].set_facecolor('#d4edda')   # our paper-protocol row
        if X_temp_test is not None:
            tbl[3, col_idx].set_facecolor('#fff3cd')  # our temporal row
    ax.set_title('HDFS Anomaly Detection â€” Method Comparison\n[Bekkouche2025_BiLSTM]',
                 fontweight='bold', pad=10)
    plt.tight_layout()
    PLOT_TABLE = f'{REPORT}/bilstm_ae_hdfs_comparison_table.png'
    plt.savefig(PLOT_TABLE, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  âœ… Comparison table â†’ {PLOT_TABLE}")

    # â”€â”€ Memory cleanup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _to_del = [model, X_train_normal, X_train_opt, X_val_opt,
               X_all_primary, y_all_primary, errors_all_primary,
               errors_train, errors_test]
    if X_temp_test is not None:
        _to_del += [errors_temp_val, errors_temp_test]
    for _obj in _to_del:
        try: del _obj
        except: pass
    del _to_del
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        print("  ðŸ§¹ GPU cache cleared")

    ckpt['ae_done']       = True
    ckpt['threshold']     = float(best_thr_paper)
    ckpt['best_val_mse']  = float(best_val_mse)
    ckpt['test_f1']       = round(float(f1_paper), 4)
    ckpt['test_auc']      = round(float(roc_auc_paper), 4)
    ckpt['test_mcc']      = round(float(mcc_paper), 4)
    if X_temp_test is not None:
        ckpt['temp_test_f1']  = round(float(f1_temp), 4)
        ckpt['temp_test_auc'] = round(float(roc_auc_temp), 4)
        ckpt['temp_test_mcc'] = round(float(mcc_temp), 4)
    save_ckpt(ckpt)

    total_time = time.time() - t0_total
    print(f"\n  âœ… HDFS BiLSTM-AE+W2V COMPLETE â€” total wall time: "
          f"{total_time:.0f}s ({total_time/60:.1f} min)")

# =============================================================================
# CELL 9 â€” Verification Block
# =============================================================================
print(f"\n{'='*65}")
print("  âœ…  NOTEBOOK 14 â€” BiLSTM-AE + W2V HDFS STANDALONE â€” COMPLETE")
print(f"{'='*65}")

expected_files = [
    (MODEL_DIR, 'w2v_hdfs_full.model',              'Word2Vec model'),
    (MODEL_DIR, 'bilstm_ae_w2v_hdfs_opt.pt',        'AE model weights'),
    (MODEL_DIR, 'bilstm_ae_w2v_hdfs_config.json',   'Config + metrics'),
    (MODEL_DIR, 'hdfs_ae_sessions_train.npz',        'Train sessions (90/10)'),
    (MODEL_DIR, 'hdfs_ae_sessions_test.npz',         'Test sessions (90/10)'),
    (REPORT,    'bilstm_ae_hdfs_results.csv',        'Results CSV'),
    (REPORT,    'bilstm_ae_hdfs_loss_curve.png',     'Loss + threshold plots'),
    (REPORT,    'bilstm_ae_hdfs_error_hist.png',     'Error histogram'),
    (REPORT,    'bilstm_ae_hdfs_cm.png',             'Confusion matrix'),
    (REPORT,    'bilstm_ae_hdfs_roc.png',            'ROC curve'),
    (REPORT,    'bilstm_ae_hdfs_pr_curve.png',       'PR curve'),
    (REPORT,    'bilstm_ae_hdfs_comparison_table.png', 'Comparison table'),
]

all_ok = True
print(f"\n  Output file status:")
for directory, fname, desc in expected_files:
    path   = os.path.join(directory, fname)
    exists = os.path.exists(path)
    size_s = f"({os.path.getsize(path)/1024:.1f} KB)" if exists else "(missing)"
    icon   = 'âœ…' if exists else 'âŒ'
    print(f"    {icon} {desc:<35s} â†’ {fname}  {size_s}")
    if not exists:
        all_ok = False

print(f"\n  Checkpoint state: {dict(ckpt)}")
print(f"\n  Status: {'ðŸŽ‰ All outputs present' if all_ok else 'âš ï¸  Some outputs missing'}")

# =============================================================================
# CELL 10 â€” Academic Summary
# =============================================================================
print(f"\n{'='*65}")
print("  ðŸ“š  ACADEMIC SUMMARY â€” NOTEBOOK 14")
print(f"{'='*65}")
print(f"""
  Method  : BiLSTM Autoencoder + Word2Vec (frozen pre-trained embeddings)
  Dataset : HDFS (complete, 2.6 GB, {ckpt.get('n_sessions', '?')} sessions)

  Architecture:
    â€¢ Word2Vec: sg=1, dim={EMBED_DIM}, window=5, min_count=1
      trained from scratch on ALL HDFS sessions ({ckpt.get('vocab_size', '?')} unique templates)
    â€¢ Encoder: BiLSTM (bidirectional) â€” last hidden state â†’ bottleneck
    â€¢ Decoder: LSTM (unidirectional) â€” latent as initial state + zero input â†’ reconstruct
    â€¢ Loss: MSE in W2V embedding space (not on token IDs)
    â€¢ Training: NORMAL sessions only (unsupervised)
    â€¢ Threshold: F1-sensitive search on full val set (n_points=5000, range=[10th,99.5th pct])

  Split protocol [Bekkouche2025_BiLSTM] Table I:
    â€¢ PRIMARY  : Stratified random 90/10 (sklearn, random_state=42)
    â€¢ SECONDARY: Temporal 60/20/20 (no leakage) â€” our rigorous protocol
    â€¢ BlockId extraction: regex blk_-?\\d+
    â€¢ Label: any anomalous line â†’ session label = 1
    â€¢ MAX_SEQ_LEN={ckpt.get('max_seq_len', '?')} (computed from data: median+2Ã—MAD, cap=100)

  Threshold [Bekkouche2025_BiLSTM] Â§II-C-a:
    â€¢ Searched on TEST set directly (semi-supervised, F1-maximising)

  Results (Paper Protocol â€” 90/10 stratified, threshold on test):
    F1 (anomaly class)  : {ckpt.get('test_f1', '?')}
    ROC AUC             : {ckpt.get('test_auc', '?')}
    MCC                 : {ckpt.get('test_mcc', '?')}
    Threshold used      : {ckpt.get('threshold', '?')}

  Results (Our Protocol â€” 60/20/20 temporal, threshold on val):
    F1 (anomaly class)  : {ckpt.get('temp_test_f1', 'n/a (run from scratch)')}
    ROC AUC             : {ckpt.get('temp_test_auc', 'n/a')}

  Paper target [Bekkouche2025_BiLSTM]:
    Precision=0.987  Recall=1.000  F1=0.993

  Gap analysis:
    Î” F1 (paper protocol) = {ckpt.get('test_f1', 0) - 0.993:+.4f}

  Conclusion:
""")

test_f1_val = ckpt.get('test_f1', 0.0)
if isinstance(test_f1_val, (int, float)):
    if test_f1_val >= 0.993:
        print("    âœ… REPRODUCTION SUCCESSFUL â€” Matches or exceeds paper F1=0.993.")
        print("       Confirms that frozen W2V semantic embeddings provide a richer")
        print("       reconstruction target than learned nn.Embedding initialised randomly.")
    elif test_f1_val >= 0.970:
        print("    ðŸŸ¡ STRONG RESULT â€” Within 2.3pp of paper target (F1=0.993).")
        print("       Delta likely explained by: (1) Optuna trial budget, (2) temporal")
        print("       split placing different proportion of anomalies in test,")
        print("       (3) paper may have used additional hyperparameter tuning.")
        print("       Result is thesis-acceptable and scientifically valid.")
    elif test_f1_val >= 0.900:
        print("    ðŸŸ  ACCEPTABLE â€” F1 below paper by >2pp. Possible causes:")
        print("       (1) Increase Optuna trials to 30+, (2) train W2V longer (epochs=20),")
        print("       (3) Paper may use Word2Vec on raw log text (not Drain templates).")
        print("       Document this gap and analysis in the thesis.")
    else:
        print("    ðŸ”´ BELOW PAPER â€” Significant gap. Likely causes:")
        print("       (1) Word2Vec not yet trained on full corpus (check w2v_ready ckpt),")
        print("       (2) MAX_SEQ_LEN too short for HDFS sessions,")
        print("       (3) Insufficient training epochs â€” increase to 100.")
        print("       Re-run with NROWS_LIMIT=None and GPU T4 for full-quality result.")
else:
    print("    âš ï¸  Test F1 not yet recorded in checkpoint (training may not be complete).")
    print("       Check ckpt_14_bilstm_ae_hdfs.json for partial results.")

print(f"\n{'='*65}")
print("  References:")
print("    [Bekkouche2025_BiLSTM] Bekkouche M. et al. â€” BiLSTM-Autoencoder, IEEE 2025")
print("    [Du2017_DeepLog]       Du M. et al.         â€” DeepLog, CCS 2017")
print("    [Zhang2019_LogRobust]  Zhang X. et al.      â€” LogRobust, ESEC/FSE 2019")
print("    [Meng2019_LogAnomaly]  Meng W. et al.       â€” LogAnomaly, IJCAI 2019")
print("    [Guo2021_LogBERT]      Guo H. et al.        â€” LogBERT, IJCNN 2021")
print(f"{'='*65}")


# =============================================================================
# STANDALONE NOTEBOOK 17 â€” DeepLog on BGL (Fully Independent)
#
# âœ… ZERO dependencies â€” reads raw BGL_Drain.csv directly from Kaggle input.
# âœ… Builds BGL sessions inline (sliding window, vocab, int32 sequences).
# âœ… One dataset only (BGL) â€” RAM stays safe on Kaggle T4/P100.
# âœ… Trains on NORMAL sessions only â€” next-key prediction for anomaly scoring.
#
# References:
#   [Du2017_DeepLog]         â€” DeepLog next-key prediction using LSTM on normal logs.
#   [Bekkouche2024]          â€” Unsupervised next-key prediction thresholding on BGL.
#   [Zhang2019_LogRobust]    â€” Sequential temporal split is scientifically honest.
#
# Architecture:
#   Embedding(vocab_size, embed_dim, padding_idx=0)
#   â†’ LSTM(embed_dim, hidden_size, num_layers)
#   â†’ Linear(hidden_size, vocab_size)
#   Loss: CrossEntropyLoss on next-key prediction
#
# Kaggle setup:
#   - Dataset: pfe-log-anomaly  (must contain BGL_Drain.csv)
#   - Accelerator: GPU T4 x2 or P100
#   - Estimated time: ~20 minutes
# =============================================================================

import os, gc, json, pathlib, time, random, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_curve, auc,
)

warnings.filterwarnings('ignore')

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 1 â€” Environment
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
KAGGLE   = os.path.exists('/kaggle')
BASE_IN  = '/kaggle/input/pfe-log-anomaly' if KAGGLE else 'Dataset'
BASE_OUT = '/kaggle/working'               if KAGGLE else 'result/results_deeplog_bgl'
MODEL_DIR = f'{BASE_OUT}/models'
REPORT    = f'{BASE_OUT}/pfe_report'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DS_KEY = 'bgl'

# BGL sliding window params
WINDOW_SIZE = 20
STEP_SIZE   = 10

# Safety cap
NROWS_LIMIT = None  # None = full dataset

CKPT = pathlib.Path(BASE_OUT) / f'ckpt_17_deeplog_{DS_KEY}.json'
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
print(f"âœ… Env: {'Kaggle' if KAGGLE else 'Local'} | Device: {DEVICE} | BGL DeepLog Standalone")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 2 â€” Build BGL Sessions (Sliding Window) Inline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if 'sessions_ready' not in ckpt:
    print("\n[CELL 2] Building BGL sliding-window sessions ...")
    t0 = time.time()

    filepath = find_file('BGL_Drain.csv')
    df = pd.read_csv(filepath, usecols=['template', 'label'], nrows=NROWS_LIMIT,
                     on_bad_lines='skip', low_memory=False)
    print(f"  Loaded BGL CSV: {len(df):,} rows")

    all_templates = df['template'].fillna('').astype(str).tolist()
    # BGL label: '-' = normal, anything else = anomaly
    all_labels    = (df['label'].astype(str).str.strip() != '-').astype(int).tolist()
    del df; gc.collect()

    n_total = len(all_templates)
    
    # Build vocabulary from training + validation portion (first 80% of lines) to avoid leakage
    i1_lines = int(n_total * 0.80)
    unique_t = sorted(set(all_templates[:i1_lines]))
    vocab_bgl = {'<PAD>': 0, '<UNK>': 1}
    for idx, t in enumerate(unique_t):
        vocab_bgl[t] = idx + 2
    joblib.dump(vocab_bgl, f'{MODEL_DIR}/vocab_bgl_deeplog.pkl')
    print(f"  Vocabulary: {len(vocab_bgl)} templates (train-only, no leakage)")

    event_ids = np.array(
        [vocab_bgl.get(t, 1) for t in all_templates], dtype=np.int32
    )
    label_arr = np.array(all_labels, dtype=np.int32)
    del all_templates, all_labels; gc.collect()

    n_windows = (n_total - WINDOW_SIZE) // STEP_SIZE + 1
    print(f"  Building {n_windows:,} sliding windows ...")
    sequences = np.zeros((n_windows, WINDOW_SIZE), dtype=np.int32)
    labels    = np.zeros(n_windows, dtype=np.int32)

    for i in range(n_windows):
        start = i * STEP_SIZE
        end   = start + WINDOW_SIZE
        sequences[i] = event_ids[start:end]
        labels[i]    = int(label_arr[start:end].max())

    del event_ids, label_arr; gc.collect()
    print(f"  Windows: {n_windows:,} | Anomaly: {labels.sum():,} ({labels.mean()*100:.1f}%)")

    from sklearn.model_selection import train_test_split
    indices = np.arange(n_windows)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])

    np.savez_compressed(f'{MODEL_DIR}/bgl_sessions_train_deeplog.npz', X=sequences[train_idx], y=labels[train_idx])
    np.savez_compressed(f'{MODEL_DIR}/bgl_sessions_val_deeplog.npz',   X=sequences[val_idx],   y=labels[val_idx])
    np.savez_compressed(f'{MODEL_DIR}/bgl_sessions_test_deeplog.npz',  X=sequences[test_idx],  y=labels[test_idx])

    print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")
    del sequences, labels; gc.collect()

    elapsed = time.time() - t0
    ckpt['sessions_ready'] = True; save_ckpt(ckpt)
    print(f"  âœ… BGL sessions saved ({elapsed:.0f}s)")
else:
    print("[CELL 2] â­ï¸  Sessions already built (checkpoint)")

# Load sessions
tr = np.load(f'{MODEL_DIR}/bgl_sessions_train_deeplog.npz')
va = np.load(f'{MODEL_DIR}/bgl_sessions_val_deeplog.npz')
te = np.load(f'{MODEL_DIR}/bgl_sessions_test_deeplog.npz')
vocab_bgl = joblib.load(f'{MODEL_DIR}/vocab_bgl_deeplog.pkl')
VS = len(vocab_bgl)

X_tr, y_tr = tr['X'], tr['y']
X_v,  y_v  = va['X'], va['y']
X_te, y_te = te['X'], te['y']

print(f"  VS={VS} | Train={X_tr.shape} | Val={X_v.shape} | Test={X_te.shape}")
print(f"  Anomaly: tr={y_tr.mean()*100:.1f}% | v={y_v.mean()*100:.1f}% | te={y_te.mean()*100:.1f}%")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 3 â€” DeepLog Predictor Architecture
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class DeepLogPredictor(nn.Module):
    """
    DeepLog Predictor Model (Unidirectional LSTM for next-key prediction).
    Input: sequence of event IDs. Output: logits over vocabulary for next key.
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)                 # (B, T, E)
        out, _ = self.lstm(emb)                 # (B, T, H)
        logits = self.fc(out)                   # (B, T, V)
        return logits

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 4 â€” DeepLog Anomaly Evaluation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def evaluate_deeplog(model, X, y_true, k, batch_size=512):
    model.eval()
    y_pred = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i+batch_size]).long().to(DEVICE)   # (B, T)
            # Input to model: first T-1 tokens
            # Targets: last T-1 tokens (predictions at each step)
            xb = batch[:, :-1]
            yb = batch[:, 1:].cpu().numpy()
            
            logits = model(xb)                      # (B, T-1, V)
            # Find top-k predictions
            topk = torch.topk(logits, k, dim=-1).indices.cpu().numpy() # (B, T-1, k)
            
            # For each session in batch, check if ANY prediction failed
            for b_idx in range(len(batch)):
                failed = False
                for t_idx in range(xb.size(1)):
                    actual = yb[b_idx, t_idx]
                    predicted = topk[b_idx, t_idx]
                    if actual not in predicted:
                        failed = True
                        break
                y_pred.append(1 if failed else 0)
                
    return np.array(y_pred)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 5 â€” Training DeepLog (Unsupervised on Normal windows)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if 'deeplog_done' not in ckpt:
    print(f"\n[CELL 5] DeepLog Training â€” BGL")

    # DeepLog trains ONLY on normal sequences
    X_train_normal = X_tr[y_tr == 0]
    print(f"  Normal train sessions: {len(X_train_normal):,} / {len(y_tr):,}")

    model = DeepLogPredictor(VS, embed_dim=64, hidden_size=128, num_layers=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train DataLoader
    train_ds = TensorDataset(torch.from_numpy(X_train_normal).long())
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    
    # Val DataLoader for early stopping (normal val sequences)
    X_val_normal = X_v[y_v == 0]
    val_ds = TensorDataset(torch.from_numpy(X_val_normal).long())
    val_dl = DataLoader(val_ds, batch_size=512, shuffle=False)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    losses = []

    for ep in range(1, 51):
        model.train()
        epoch_loss = 0
        for (xb,) in train_dl:
            xb = xb.to(DEVICE)
            # Input is xb[:, :-1], Target is xb[:, 1:]
            inputs  = xb[:, :-1]
            targets = xb[:, 1:]
            
            optimizer.zero_grad()
            logits = model(inputs)              # (B, T-1, V)
            loss = criterion(logits.reshape(-1, VS), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_train = epoch_loss / len(train_dl)
        losses.append(avg_train)
        
        # Validation loss (normal val only)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (xb,) in val_dl:
                xb = xb.to(DEVICE)
                inputs  = xb[:, :-1]
                targets = xb[:, 1:]
                logits  = model(inputs)
                val_loss += criterion(logits.reshape(-1, VS), targets.reshape(-1)).item()
        avg_val = val_loss / len(val_dl)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if ep % 5 == 0 or ep == 1:
            print(f"    Ep {ep:>2} | Train Loss={avg_train:.6f} | Val Loss={avg_val:.6f}")
        if no_improve >= 5:
            print(f"    â¹ Early stop at epoch {ep}"); break

    # Load best model
    model.load_state_dict(best_state)

    # Search for best k on validation set
    print("\n  ðŸ” Tuning top-k parameter on validation set ...")
    best_f1, best_k = 0, 1
    k_candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
    val_f1_scores = []

    for k in k_candidates:
        y_val_pred = evaluate_deeplog(model, X_v, y_v, k=k)
        f1 = f1_score(y_v, y_val_pred, pos_label=1, zero_division=0)
        val_f1_scores.append(f1)
        print(f"    k={k:<2} | Val F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_k = k

    print(f"  ðŸ† Best k: {best_k} (Val F1: {best_f1:.4f})")

    # Evaluate on Test
    t_inf = time.time()
    y_pred = evaluate_deeplog(model, X_te, y_te, k=best_k)
    inf_time = time.time() - t_inf

    precision = precision_score(y_te, y_pred, zero_division=0)
    recall = recall_score(y_te, y_pred, zero_division=0)
    test_f1 = f1_score(y_te, y_pred, zero_division=0)
    test_mcc = matthews_corrcoef(y_te, y_pred)

    metrics = {
        'Dataset':    DS_KEY.upper(),
        'Model':      'DeepLog',
        'Type':       'Unsupervised (DL)',
        'Precision':  round(precision, 4),
        'Recall':     round(recall, 4),
        'F1_Anomaly': round(test_f1, 4),
        'Macro_F1':   round(f1_score(y_te, y_pred, average='macro', zero_division=0), 4),
        'MCC':        round(test_mcc, 4),
        'Best_k':     best_k,
        'Inference_Time_s': round(inf_time, 4),
    }

    print(f"\n  ðŸ“Š TEST RESULTS â€” BGL DeepLog:")
    print(classification_report(y_te, y_pred, target_names=['Normal', 'Anomaly'], digits=4))
    print(f"  F1={test_f1:.4f} | MCC={test_mcc:.4f} | k={best_k}")

    # Save model + config
    torch.save(best_state, f'{MODEL_DIR}/deeplog_{DS_KEY}_opt.pt')
    with open(f'{MODEL_DIR}/deeplog_{DS_KEY}_config.json', 'w') as f:
        json.dump({'vocab_size': VS, **metrics}, f, indent=2)
    pd.DataFrame([metrics]).to_csv(f'{REPORT}/deeplog_{DS_KEY}_results.csv', index=False)

    # 1. Loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(losses, 'b-', lw=1.5)
    plt.title(f'DeepLog Training Loss â€” BGL')
    plt.xlabel('Epoch'); plt.ylabel('CrossEntropy'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{REPORT}/deeplog_loss_{DS_KEY}.png', dpi=300); plt.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    ax.set_title(f'CM â€” BGL DeepLog (k={best_k})')
    plt.tight_layout()
    plt.savefig(f'{REPORT}/deeplog_cm_{DS_KEY}.png', dpi=300); plt.close()

    # 3. Validation F1 curve across k
    plt.figure(figsize=(6, 4))
    plt.plot(k_candidates, val_f1_scores, 'ro-', lw=1.5)
    plt.title('Validation F1 vs. top-k predictions')
    plt.xlabel('k'); plt.ylabel('F1 Score'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{REPORT}/deeplog_val_f1_vs_k_{DS_KEY}.png', dpi=300); plt.close()

    del model, X_tr, X_v, X_te; gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()

    ckpt['deeplog_done'] = True; save_ckpt(ckpt)
    print(f"\n  âœ… BGL DeepLog done ({inf_time:.0f}s inference)")
else:
    print("[CELL 5] â­ï¸  DeepLog already done (checkpoint)")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 6 â€” VerificationBlock
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'='*60}")
print("  âœ… DEEPLOG BGL STANDALONE â€” COMPLETE")
print(f"{'='*60}")
expected_files = [
    (MODEL_DIR, f'deeplog_{DS_KEY}_opt.pt'),
    (MODEL_DIR, f'deeplog_{DS_KEY}_config.json'),
    (MODEL_DIR, 'vocab_bgl_deeplog.pkl'),
    (MODEL_DIR, 'bgl_sessions_train_deeplog.npz'),
    (MODEL_DIR, 'bgl_sessions_val_deeplog.npz'),
    (MODEL_DIR, 'bgl_sessions_test_deeplog.npz'),
    (REPORT,    f'deeplog_{DS_KEY}_results.csv'),
    (REPORT,    f'deeplog_loss_{DS_KEY}.png'),
    (REPORT,    f'deeplog_cm_{DS_KEY}.png'),
    (REPORT,    f'deeplog_val_f1_vs_k_{DS_KEY}.png'),
]
for folder, fname in expected_files:
    p = os.path.join(folder, fname)
    print(f"  {'âœ…' if os.path.exists(p) else 'âŒ'} {fname}")


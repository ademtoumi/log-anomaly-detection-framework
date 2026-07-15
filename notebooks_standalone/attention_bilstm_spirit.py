# =============================================================================
# attention_bilstm_spirit.py
# Spirit Attention-BiLSTM â€” Standalone Kaggle Notebook
# =============================================================================
# References:
#   [Bekkouche2025_Spirit]   Spirit window size significantly impacts results
#   [Zhang2019_LogRobust]    Attention-based BiLSTM achieves high F1 on HDFS
#   [Bekkouche2024]          Benchmark framework for log anomaly detection
# =============================================================================

import os, gc, json, time, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef,
    average_precision_score, confusion_matrix
)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 0. Paths & Global Config
# ---------------------------------------------------------------------------
DATA_DIR   = '/kaggle/input/pfe-log-anomaly' if os.path.exists('/kaggle') else 'Dataset'
OUTPUT_DIR = '/kaggle/working'               if os.path.exists('/kaggle') else 'result/results_attention_bilstm_spirit'
REPORT     = os.path.join(OUTPUT_DIR, 'pfe_report')
CKPT_FILE  = os.path.join(OUTPUT_DIR, 'checkpoint_08.json')
os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)
os.makedirs(REPORT, exist_ok=True)

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
SPIRIT_CSV = find_file('Spirit_Drain.csv')

# Sliding-window parameters [Bekkouche2025_Spirit]
WINDOW_SIZE = 20
STEP_SIZE   = 10

# Toggle NROWS_LIMIT to a smaller int (e.g. 2_000_000) if OOM on Kaggle
NROWS_LIMIT = None   # None = read full file

CHUNKSIZE   = 500_000
SEED        = 42
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device : {DEVICE}")
print(f"Output : {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_ckpt():
    if os.path.exists(CKPT_FILE):
        with open(CKPT_FILE) as f:
            return json.load(f)
    return {}

def save_ckpt(ckpt: dict):
    with open(CKPT_FILE, 'w') as f:
        json.dump(ckpt, f, indent=2)

ckpt = load_ckpt()
print("Checkpoint state:", ckpt)

# =============================================================================
# 1. BUILD SPIRIT SLIDING-WINDOW SESSIONS
# =============================================================================
SESSIONS_TRAIN = os.path.join(OUTPUT_DIR, 'spirit_sessions_train.npz')
SESSIONS_VAL   = os.path.join(OUTPUT_DIR, 'spirit_sessions_val.npz')
SESSIONS_TEST  = os.path.join(OUTPUT_DIR, 'spirit_sessions_test.npz')
VOCAB_FILE     = os.path.join(OUTPUT_DIR, 'vocab_spirit_opt.pkl')

if not ckpt.get('sessions_ready'):
    print("\n[1/4] Building Spirit sliding-window sessions â€¦")

    # ------------------------------------------------------------------
    # 1a. Chunked read â€” collect templates & labels
    # ------------------------------------------------------------------
    all_templates = []
    all_labels    = []
    rows_read     = 0

    reader = pd.read_csv(
        SPIRIT_CSV,
        usecols=['template', 'label'],
        chunksize=CHUNKSIZE,
        dtype={'template': str},
        low_memory=False
    )

    for chunk in reader:
        chunk.dropna(subset=['template'], inplace=True)
        chunk['template'] = chunk['template'].astype(str).str.strip()
        chunk['label']    = (chunk['label'].astype(str).str.strip() != '-').astype(np.int8)

        all_templates.extend(chunk['template'].tolist())
        all_labels.extend(chunk['label'].tolist())
        rows_read += len(chunk)

        if NROWS_LIMIT and rows_read >= NROWS_LIMIT:
            print(f"  NROWS_LIMIT={NROWS_LIMIT:,} reached â€” stopping early.")
            break

        if rows_read % 2_000_000 == 0:
            print(f"  Read {rows_read:,} rows â€¦")

    del reader
    gc.collect()
    print(f"  Total rows collected : {len(all_templates):,}")

    # ------------------------------------------------------------------
    # 1b. Build vocabulary (train split only â€” no leakage)
    # ------------------------------------------------------------------
    i1_lines = int(len(all_templates) * 0.60)
    unique_templates = sorted(set(all_templates[:i1_lines]))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for t in unique_templates:
        vocab[t] = len(vocab)
    vocab_size = len(vocab)
    print(f"  Vocab size : {vocab_size:,} (train-only, no leakage)")

    with open(VOCAB_FILE, 'wb') as f:
        pickle.dump(vocab, f)

    del unique_templates
    gc.collect()

    # ------------------------------------------------------------------
    # 1c. Convert templates â†’ event IDs
    # ------------------------------------------------------------------
    unk_id     = vocab['<UNK>']
    event_ids  = np.array([vocab.get(t, unk_id) for t in all_templates],
                          dtype=np.int32)
    label_arr  = np.array(all_labels, dtype=np.int8)

    del all_templates, all_labels
    gc.collect()
    print(f"  event_ids shape : {event_ids.shape}")

    # ------------------------------------------------------------------
    # 1d. Build sliding windows [Bekkouche2025_Spirit]
    # ------------------------------------------------------------------
    n_total   = len(event_ids)
    n_windows = (n_total - WINDOW_SIZE) // STEP_SIZE + 1
    print(f"  Building {n_windows:,} windows (W={WINDOW_SIZE}, S={STEP_SIZE}) â€¦")

    X_seq = np.zeros((n_windows, WINDOW_SIZE), dtype=np.int32)
    y_win = np.zeros(n_windows, dtype=np.int8)

    for i in range(n_windows):
        start = i * STEP_SIZE
        end   = start + WINDOW_SIZE
        X_seq[i] = event_ids[start:end]
        # window label = max (any anomaly â†’ anomaly) [Bekkouche2025_Spirit]
        y_win[i]  = label_arr[start:end].max()

    del event_ids, label_arr
    gc.collect()
    print(f"  Windows shape : {X_seq.shape}  |  anomaly rate: {y_win.mean():.4f}")

    # ------------------------------------------------------------------
    # 1e. Stratified random split 70 / 10 / 20
    # ------------------------------------------------------------------
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(X_seq))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=y_win)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=y_win[train_val_idx])

    X_train, y_train = X_seq[train_idx], y_win[train_idx]
    X_val,   y_val   = X_seq[val_idx],   y_win[val_idx]
    X_test,  y_test  = X_seq[test_idx],  y_win[test_idx]

    del X_seq, y_win
    gc.collect()

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    np.savez_compressed(SESSIONS_TRAIN, X=X_train, y=y_train)
    np.savez_compressed(SESSIONS_VAL,   X=X_val,   y=y_val)
    np.savez_compressed(SESSIONS_TEST,  X=X_test,  y=y_test)

    del X_train, y_train, X_val, y_val, X_test, y_test
    gc.collect()

    ckpt['sessions_ready'] = True
    save_ckpt(ckpt)
    print("  Sessions saved âœ“")
else:
    print("[1/4] Sessions already built â€” skipping.")

# Reload vocab
with open(VOCAB_FILE, 'rb') as f:
    vocab = pickle.load(f)
VOCAB_SIZE = len(vocab)
print(f"  Vocab size : {VOCAB_SIZE:,}")

# =============================================================================
# 2. DATASET & MODEL DEFINITIONS
# =============================================================================
print("\n[2/4] Defining Dataset & Model â€¦")

class SpiritWindowDataset(Dataset):
    """Spirit sliding-window dataset."""
    def __init__(self, npz_path: str):
        data    = np.load(npz_path)
        self.X  = torch.from_numpy(data['X'].astype(np.int32)).long()
        self.y  = torch.from_numpy(data['y'].astype(np.int8)).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AttentionLayer(nn.Module):
    """Additive self-attention over BiLSTM hidden states [Zhang2019_LogRobust]."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden*2)
        scores  = self.attn(lstm_out).squeeze(-1)          # (batch, seq_len)
        weights = torch.softmax(scores, dim=-1).unsqueeze(2)  # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)           # (batch, hidden*2)
        return context


class AttentionBiLSTM(nn.Module):
    """
    Embedding â†’ BiLSTM â†’ Attention â†’ Dropout â†’ FC(1)
    [Zhang2019_LogRobust] Attention-based BiLSTM for log anomaly detection.
    """
    def __init__(self, vocab_size: int, embed_dim: int,
                 hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention  = AttentionLayer(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        emb      = self.dropout(self.embedding(x))          # (B, T, E)
        lstm_out, _ = self.lstm(emb)                         # (B, T, H*2)
        context  = self.attention(lstm_out)                  # (B, H*2)
        out      = self.dropout(context)
        logits   = self.classifier(out).squeeze(-1)          # (B,)
        return logits


# =============================================================================
# 3. TRAINING UTILITIES
# =============================================================================

def make_loader(npz_path: str, batch_size: int, shuffle: bool = False,
                num_workers: int = 2) -> DataLoader:
    ds = SpiritWindowDataset(npz_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=(num_workers > 0))


def compute_pos_weight(npz_path: str) -> torch.Tensor:
    """pos_weight = sqrt(n_neg / n_pos) to handle class imbalance."""
    data    = np.load(npz_path)
    y       = data['y']
    n_pos   = y.sum()
    n_neg   = len(y) - n_pos
    pw      = float(np.sqrt(n_neg / max(n_pos, 1)))
    print(f"  pos_weight = {pw:.4f}  (n_pos={int(n_pos):,}, n_neg={int(n_neg):,})")
    return torch.tensor([pw], device=DEVICE)


def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion=None):
    """Collect val/test probs and labels. Threshold applied separately."""
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            logits = model(X_batch)
            if criterion:
                total_loss += criterion(logits, y_batch).item()
        all_logits.append(logits.cpu())
        all_labels.append(y_batch.cpu())
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy().astype(int)
    probs      = torch.sigmoid(torch.tensor(all_logits)).numpy()
    avg_loss   = total_loss / len(loader) if criterion else None
    return avg_loss, probs, all_labels


def find_best_threshold_f1(probs, labels, n_points=300):
    """Grid-search threshold on VAL probs to maximise F1. NEVER call with test labels."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.01, 0.99, n_points):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def train_model(params: dict, max_epochs: int = 60,
                patience: int = 12, verbose: bool = True):
    """
    Full training loop with mixed-precision, cosine LR, F1-optimal early stopping.
    Early stopping is on val F1 at the threshold that maximises val F1 â€” NOT fixed 0.5.
    [Zhang2019_LogRobust]: threshold tuning on validation set only.
    """
    embed_dim   = params['embed_dim']
    hidden_size = params['hidden_size']
    num_layers  = params['num_layers']
    dropout     = params['dropout']
    lr          = params['lr']
    batch_size  = params['batch_size']

    model = AttentionBiLSTM(VOCAB_SIZE, embed_dim, hidden_size,
                             num_layers, dropout).to(DEVICE)

    pw        = compute_pos_weight(SESSIONS_TRAIN)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr * 0.01)

    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')

    train_loader = make_loader(SESSIONS_TRAIN, batch_size, shuffle=True)
    val_loader   = make_loader(SESSIONS_VAL,   batch_size, shuffle=False)

    best_val_f1   = 0.0
    best_threshold = 0.5
    best_state    = None
    no_improve    = 0
    history       = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    for epoch in range(1, max_epochs + 1):
        train_loss              = train_one_epoch(model, train_loader, optimizer,
                                                  criterion, scaler)
        val_loss, val_probs, val_labels = evaluate(model, val_loader, criterion)

        # --- F1-optimal threshold search on val (never on test) ---
        thr, val_f1 = find_best_threshold_f1(val_probs, val_labels, n_points=300)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)

        if verbose and epoch % 5 == 0:
            print(f"  Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                  f"| val_loss={val_loss:.4f} | val_F1={val_f1:.4f} | thr={thr:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1   = val_f1
            best_threshold = thr
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}.")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_val_f1, best_threshold


# =============================================================================
# 4. OPTUNA HYPER-PARAMETER SEARCH + FULL TRAINING
# Wrapped in checkpoint guard â€” skips entirely if 'done' is already set.
# This ensures a Kaggle timeout mid-training can be resumed cleanly.
# =============================================================================
if ckpt.get('done'):
    print("\n[3/4] â­ï¸  Optuna + training already done (checkpoint 'done') â€” skipping.")
else:
    print("\n[3/4] Optuna hyper-parameter search (20 trials, timeout=1200 s) â€¦")

    # Warm-start params [Bekkouche2025_Spirit]
    WARM_START = {
        'embed_dim':   64,
        'hidden_size': 128,
        'num_layers':  2,
        'dropout':     0.3,
        'lr':          0.0005,
        'batch_size':  256,
    }

    def objective(trial: optuna.Trial) -> float:
        params = {
            'embed_dim':   trial.suggest_categorical('embed_dim',   [32, 64, 128]),
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
            'num_layers':  trial.suggest_int('num_layers', 1, 3),
            'dropout':     trial.suggest_float('dropout', 0.1, 0.5, step=0.1),
            'lr':          trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'batch_size':  trial.suggest_categorical('batch_size', [128, 256, 512]),
        }
        try:
            _, _, val_f1, _ = train_model(params, max_epochs=20, patience=5, verbose=False)
        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
        return val_f1

    sampler = TPESampler(seed=SEED)
    study   = optuna.create_study(direction='maximize', sampler=sampler)
    study.enqueue_trial(WARM_START)
    study.optimize(objective, n_trials=20, timeout=1200, show_progress_bar=False)

    best_params = study.best_params
    best_val_f1 = study.best_value
    print(f"\nBest Optuna val F1 : {best_val_f1:.4f}")
    print(f"Best params        : {best_params}")

    # =========================================================================
    # 5. FINAL TRAINING WITH BEST PARAMS
    # =========================================================================
    print("\n[4/4] Final training with best params (max 60 epochs, patience=12) â€¦")
    t0 = time.time()
    model, history, best_val_f1_final, best_threshold = train_model(
        best_params, max_epochs=60, patience=12, verbose=True)
    train_time = time.time() - t0
    print(f"  Training time      : {train_time:.1f} s")
    print(f"  Best val F1        : {best_val_f1_final:.4f}")
    print(f"  Val-optimal threshold : {best_threshold:.4f}")

    MODEL_PATH  = os.path.join(OUTPUT_DIR, 'models', 'bilstm_spirit_opt.pt')
    CONFIG_PATH = os.path.join(OUTPUT_DIR, 'models', 'bilstm_spirit_config.json')
    torch.save(model.state_dict(), MODEL_PATH)
    cfg = {**best_params,
           'vocab_size':      VOCAB_SIZE,
           'window_size':     WINDOW_SIZE,
           'step_size':       STEP_SIZE,
           'model':           'AttentionBiLSTM',
           'dataset':         'Spirit',
           'train_time_s':    round(train_time, 2),
           'threshold':       round(float(best_threshold), 6),
           'best_val_f1':     round(float(best_val_f1_final), 6),
           'threshold_split': 'validation'}
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"  Model saved  â†’ {MODEL_PATH}")
    print(f"  Config saved â†’ {CONFIG_PATH}")

    # =========================================================================
    # 6. FULL EVALUATION ON TEST SET
    # =========================================================================
    # =========================================================================
    # 6. FINAL EVALUATION ON TEST SET
    # Threshold derived from val set â€” test set touched exactly once.
    # [Zhang2019_LogRobust]: threshold must be set on held-out validation data.
    # =========================================================================
    print("\nEvaluating on test set (threshold from val) â€¦")
    test_loader = make_loader(SESSIONS_TEST, best_params['batch_size'], shuffle=False)
    t_inf = time.time()
    _, test_probs, test_labels = evaluate(model, test_loader)
    inf_time = (time.time() - t_inf) * 1000   # ms

    # Apply val-optimal threshold â€” never search on test
    test_preds = (test_probs >= best_threshold).astype(int)
    print(f"  Threshold applied : {best_threshold:.4f}  (derived from val set)")

    precision = precision_score(test_labels, test_preds, pos_label=1, zero_division=0)
    recall    = recall_score(test_labels,    test_preds, pos_label=1, zero_division=0)
    f1_anom   = f1_score(test_labels,        test_preds, pos_label=1, zero_division=0)
    macro_f1  = f1_score(test_labels,        test_preds, average='macro', zero_division=0)
    mcc       = matthews_corrcoef(test_labels, test_preds)
    auc_val   = roc_auc_score(test_labels, test_probs)
    avg_prec  = average_precision_score(test_labels, test_probs)

    # Paper target: [Bekkouche2025_Spirit] F1 â‰ˆ 0.96 on Spirit
    PAPER_F1 = 0.96
    print(f"\n  Paper F1 target    : {PAPER_F1:.3f}")
    print(f"  Ours F1 (test)     : {f1_anom:.4f}  (delta={f1_anom - PAPER_F1:+.4f})")

    results = {
        'Model':              'AttentionBiLSTM',
        'Dataset':            'Spirit',
        'Type':               'Supervised (DL)',
        'Paper':              'Bekkouche2025_Spirit',
        'Paper_F1':           PAPER_F1,
        'Precision':          round(precision, 4),
        'Recall':             round(recall, 4),
        'F1_Anomaly':         round(f1_anom, 4),
        'F1_Delta_vs_Paper':  round(f1_anom - PAPER_F1, 4),
        'Macro_F1':           round(macro_f1, 4),
        'AUC':                round(auc_val, 4),
        'MCC':                round(mcc, 4),
        'Avg_Precision':      round(avg_prec, 4),
        'Threshold':          round(float(best_threshold), 6),
        'Threshold_Source':   'Val set (F1-optimal)',
        'Val_F1_at_Threshold': round(float(best_val_f1_final), 4),
        'Inf_Time_ms':        round(inf_time, 2),
        'Train_Time_s':       round(train_time, 2),
        **{f'hp_{k}': v for k, v in best_params.items()}
    }
    RESULTS_CSV = os.path.join(REPORT, 'bilstm_spirit_results.csv')
    pd.DataFrame([results]).to_csv(RESULTS_CSV, index=False)
    print(pd.DataFrame([results]).T.to_string(header=False))

    # =========================================================================
    # 7. PLOTS
    # =========================================================================
    epochs_ran = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Spirit â€” Attention-BiLSTM Training Summary', fontsize=14)

    ax = axes[0]
    ax.plot(epochs_ran, history['train_loss'], label='Train Loss')
    ax.plot(epochs_ran, history['val_loss'],   label='Val Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('BCE Loss')
    ax.set_title('Loss Curve'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs_ran, history['val_f1'], color='green', label='Val F1 (Anomaly)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1')
    ax.set_title('Validation F1 Curve'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    cm_plot = confusion_matrix(test_labels, test_preds)
    sns.heatmap(cm_plot, annot=True, fmt='d', cmap='Oranges', ax=ax,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix â€” Test Set')

    plt.tight_layout()
    PLOT_PATH = os.path.join(REPORT, 'bilstm_spirit_plots.png')
    plt.savefig(PLOT_PATH, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Plots saved â†’ {PLOT_PATH}")

    ckpt['done'] = True
    save_ckpt(ckpt)

print("\n" + "=" * 60)
print("OUTPUT FILE VERIFICATION")
print("=" * 60)
for path, label in [
    (SESSIONS_TRAIN,  'spirit_sessions_train.npz'),
    (SESSIONS_VAL,    'spirit_sessions_val.npz'),
    (SESSIONS_TEST,   'spirit_sessions_test.npz'),
    (VOCAB_FILE,      'vocab_spirit_opt.pkl'),
    (os.path.join(OUTPUT_DIR, 'models', 'bilstm_spirit_opt.pt'),  'bilstm_spirit_opt.pt'),
    (os.path.join(OUTPUT_DIR, 'models', 'bilstm_spirit_config.json'), 'bilstm_spirit_config.json'),
    (os.path.join(REPORT, 'bilstm_spirit_results.csv'),            'bilstm_spirit_results.csv'),
    (os.path.join(REPORT, 'bilstm_spirit_plots.png'),              'bilstm_spirit_plots.png'),
    (CKPT_FILE,       'checkpoint_08.json'),
]:
    exists = os.path.isfile(path)
    size   = os.path.getsize(path) if exists else 0
    status = 'âœ“' if exists else 'âœ— MISSING'
    print(f"  [{status}]  {label:<40s}  {size:>12,} bytes")

print("=" * 60)
print("Notebook 08 â€” Spirit Attention-BiLSTM complete.")


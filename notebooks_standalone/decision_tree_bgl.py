#!/usr/bin/env python3
# =============================================================================
# decision_tree_bgl.py
# Decision Tree + TF-IDF on BGL dataset  (fully standalone)
# =============================================================================
# Papers:
#   [Bekkouche2025_Spirit] DT + TF-IDF achieved F1 = 0.973 on Spirit dataset
#   [Lundberg2017_SHAP]    SHAP TreeExplainer for interpretability
# =============================================================================

import os, gc, json, time, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score,
                             precision_score, recall_score)
from scipy.sparse import issparse
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR   = '/kaggle/input/pfe-log-anomaly' if os.path.exists('/kaggle') else 'Dataset'
OUTPUT_DIR = '/kaggle/working' if os.path.exists('/kaggle') else 'result/results_decision_tree_bgl'
DS_KEY     = 'bgl'
CSV_FILE   = 'BGL_Drain.csv'
REPORT     = os.path.join(OUTPUT_DIR, 'pfe_report')

TFIDF_PARAMS = dict(
    max_features  = 10_000,
    ngram_range   = (1, 3),
    sublinear_tf  = True,
    min_df        = 2,
    token_pattern = r'[a-zA-Z_:\-\.]+',
)

# Optuna
N_TRIALS   = 20
TIMEOUT    = 300          # seconds

# Warm-start (first trial anchored at these values)
WARM_START = dict(
    max_depth         = 15,
    criterion         = 'gini',
    min_samples_split = 2,
    min_samples_leaf  = 1,
    class_weight      = 'balanced',
)

# Checkpoint file
CKPT_FILE  = os.path.join(OUTPUT_DIR, f'ckpt_{DS_KEY}_dt.json')

# Output artefacts
MODEL_DIR  = os.path.join(OUTPUT_DIR, 'models')
PKL_OUT    = os.path.join(MODEL_DIR, f'dt_{DS_KEY}_opt.pkl')
CFG_OUT    = os.path.join(REPORT, f'dt_{DS_KEY}_config.json')
RES_OUT    = os.path.join(REPORT, f'dt_{DS_KEY}_results.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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
print(f"{'='*60}")
print(f"  Decision Tree â€“ BGL (standalone)")
print(f"{'='*60}")

# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================
def load_ckpt():
    if os.path.exists(CKPT_FILE):
        with open(CKPT_FILE) as f:
            return json.load(f)
    return {}

def save_ckpt(ckpt):
    with open(CKPT_FILE, 'w') as f:
        json.dump(ckpt, f, indent=2)

ckpt = load_ckpt()

# =============================================================================
# STEP 1 â€“ LOAD + SPLIT + TF-IDF
# =============================================================================
if 'data_ready' not in ckpt:
    t0 = time.time()
    print('\n[1/4] Loading BGL_Drain.csv â€¦')

    csv_path = find_file(CSV_FILE)
    df = pd.read_csv(csv_path, usecols=['template', 'label'])
    df['template'] = df['template'].fillna('').astype(str)
    df['label']    = (df['label'] != '-').astype(np.int8)   # '-' = normal

    n_total = len(df)
    n_anom  = df['label'].sum()
    print(f"   Rows: {n_total:,}  |  Anomalies: {n_anom:,} ({100*n_anom/n_total:.2f}%)")

    # Stratified random split 70/10/20 (80/20 trainval vs test)
    from sklearn.model_selection import train_test_split
    indices = np.arange(n_total)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=df['label'].values)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=df['label'].values[train_val_idx])

    X_raw_train = df['template'].iloc[train_idx].tolist()
    y_train     = df['label'].iloc[train_idx].values.astype(np.int8)

    X_raw_val   = df['template'].iloc[val_idx].tolist()
    y_val       = df['label'].iloc[val_idx].values.astype(np.int8)

    X_raw_test  = df['template'].iloc[test_idx].tolist()
    y_test      = df['label'].iloc[test_idx].values.astype(np.int8)

    del df; gc.collect()

    print(f"   Train {len(y_train):,} | Val {len(y_val):,} | Test {len(y_test):,}")

    # TF-IDF â€“ fit on TRAIN only (no leakage)
    print('[2/4] Fitting TF-IDF on train â€¦')
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = tfidf.fit_transform(X_raw_train).astype(np.float32)
    X_val   = tfidf.transform(X_raw_val).astype(np.float32)
    X_test  = tfidf.transform(X_raw_test).astype(np.float32)

    del X_raw_train, X_raw_val, X_raw_test; gc.collect()

    print(f"   TF-IDF shape: {X_train.shape}  (vocab={len(tfidf.vocabulary_):,})")
    print(f"   Elapsed: {time.time()-t0:.1f}s")

    ckpt['data_ready'] = True
    ckpt['shapes'] = {
        'train': list(X_train.shape),
        'val':   list(X_val.shape),
        'test':  list(X_test.shape),
    }
    save_ckpt(ckpt)
else:
    print('[1-2/4] Data already prepared â€“ reloading â€¦')
    # We must rebuild because sparse matrices are not persisted in the ckpt
    # (ckpt only stores flags; actual data must be re-derived from CSV)
    csv_path = find_file(CSV_FILE)
    df = pd.read_csv(csv_path, usecols=['template', 'label'])
    df['template'] = df['template'].fillna('').astype(str)
    df['label']    = (df['label'] != '-').astype(np.int8)
    n_total = len(df)
    i60 = int(0.60 * n_total)
    i80 = int(0.80 * n_total)
    X_raw_train = df['template'].iloc[:i60].tolist()
    y_train     = df['label'].iloc[:i60].values.astype(np.int8)
    X_raw_val   = df['template'].iloc[i60:i80].tolist()
    y_val       = df['label'].iloc[i60:i80].values.astype(np.int8)
    X_raw_test  = df['template'].iloc[i80:].tolist()
    y_test      = df['label'].iloc[i80:].values.astype(np.int8)
    del df; gc.collect()
    tfidf   = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = tfidf.fit_transform(X_raw_train).astype(np.float32)
    X_val   = tfidf.transform(X_raw_val).astype(np.float32)
    X_test  = tfidf.transform(X_raw_test).astype(np.float32)
    del X_raw_train, X_raw_val, X_raw_test; gc.collect()
    print(f"   Reloaded â€“ train {X_train.shape}")

# =============================================================================
# STEP 3 â€“ OPTUNA HYPERPARAMETER SEARCH
# =============================================================================
def find_best_threshold_f1(probs, labels, n_points=300):
    """Grid-search threshold on probabilities [0, 1] to maximize F1."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.01, 0.99, n_points):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

if 'dt_done' not in ckpt:
    print('\n[3/4] Optuna search (20 trials, timeout=300s) â€¦')

    def objective(trial):
        # First trial: anchor at warm-start values
        if trial.number == 0:
            max_depth         = WARM_START['max_depth']
            criterion         = WARM_START['criterion']
            min_samples_split = WARM_START['min_samples_split']
            min_samples_leaf  = WARM_START['min_samples_leaf']
            class_weight      = WARM_START['class_weight']
        else:
            max_depth         = trial.suggest_int('max_depth', 3, 40)
            criterion         = trial.suggest_categorical('criterion', ['gini', 'entropy'])
            min_samples_split = trial.suggest_int('min_samples_split', 2, 50)
            min_samples_leaf  = trial.suggest_int('min_samples_leaf', 1, 20)
            class_weight      = trial.suggest_categorical('class_weight', ['balanced', None])

        clf = DecisionTreeClassifier(
            max_depth         = max_depth,
            criterion         = criterion,
            min_samples_split = min_samples_split,
            min_samples_leaf  = min_samples_leaf,
            class_weight      = class_weight,
            random_state      = 42,
        )
        clf.fit(X_train, y_train)
        val_probs = clf.predict_proba(X_val)[:, 1]
        _, val_f1 = find_best_threshold_f1(val_probs, y_val, n_points=200)
        return val_f1

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)

    best_params = study.best_params
    best_val_f1 = study.best_value
    print(f"   Best Val F1: {best_val_f1:.4f}")
    print(f"   Best params: {best_params}")

    # Retrain on train with best params
    best_params_full = {**WARM_START, **best_params}   # merge warm-start defaults
    final_clf = DecisionTreeClassifier(**best_params_full, random_state=42)
    final_clf.fit(X_train, y_train)

    # Find optimal threshold on validation set (never touch test labels)
    val_probs = final_clf.predict_proba(X_val)[:, 1]
    best_thr, best_val_f1 = find_best_threshold_f1(val_probs, y_val, n_points=1000)

    # Test evaluation
    t_inf = time.time()
    y_prob = final_clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_thr).astype(int)
    infer_time = time.time() - t_inf

    test_f1     = f1_score(y_test, y_pred, zero_division=0)
    test_prec   = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    test_auc    = roc_auc_score(y_test, y_prob)

    print(f"\n   TEST  F1={test_f1:.4f}  Prec={test_prec:.4f}  Rec={test_recall:.4f}  AUC={test_auc:.4f}  Threshold={best_thr:.4f}")

    # Save model + config
    with open(PKL_OUT, 'wb') as f:
        pickle.dump({'model': final_clf, 'tfidf': tfidf}, f)

    cfg = {**best_params_full,
           'val_f1': best_val_f1,
           'test_f1': test_f1,
           'test_precision': test_prec,
           'test_recall': test_recall,
           'test_auc': test_auc,
           'threshold': best_thr}
    with open(CFG_OUT, 'w') as f:
        json.dump(cfg, f, indent=2)

    results_df = pd.DataFrame([cfg])
    results_df.to_csv(RES_OUT, index=False)

    ckpt['dt_done']      = True
    ckpt['test_f1']      = float(test_f1)
    ckpt['test_auc']     = float(test_auc)
    ckpt['best_params']  = best_params_full
    save_ckpt(ckpt)

else:
    print('[3/4] DT already trained â€“ loading from checkpoint â€¦')
    with open(PKL_OUT, 'rb') as f:
        bundle = pickle.load(f)
    final_clf   = bundle['model']
    y_pred      = final_clf.predict(X_test)
    y_prob      = final_clf.predict_proba(X_test)[:, 1]
    test_f1     = ckpt.get('test_f1', 0.0)
    test_auc    = ckpt.get('test_auc', 0.0)
    best_params_full = ckpt.get('best_params', WARM_START)

# =============================================================================
# STEP 4 â€“ PLOTS + SHAP
# =============================================================================
print('\n[4/4] Generating plots â€¦')

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Normal', 'Anomaly']); ax.set_yticklabels(['Normal', 'Anomaly'])
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'DT BGL â€“ Confusion Matrix\nF1={test_f1:.4f}')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(REPORT, f'dt_{DS_KEY}_cm.png'), dpi=150)
plt.close()

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, label=f'DT (AUC={test_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC â€“ DT BGL')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORT, f'dt_{DS_KEY}_roc.png'), dpi=150)
plt.close()

# --- Feature Importance Bar ---
feat_names = tfidf.get_feature_names_out()
importances = final_clf.feature_importances_
top_k = 20
idx = np.argsort(importances)[::-1][:top_k]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(feat_names[idx][::-1], importances[idx][::-1])
ax.set_xlabel('Importance')
ax.set_title(f'Top-{top_k} DT Feature Importances â€“ BGL')
plt.tight_layout()
plt.savefig(os.path.join(REPORT, f'dt_{DS_KEY}_feat_imp.png'), dpi=150)
plt.close()

# --- SHAP TreeExplainer  [Lundberg2017_SHAP] ---
try:
    import shap
    from scipy.sparse import issparse
    # Convert sparse to dense for SHAP (sample 500 test points to keep RAM low)
    n_shap = min(500, X_test.shape[0])
    rng = np.random.default_rng(42)
    idx_shap = rng.choice(X_test.shape[0], n_shap, replace=False)
    if issparse(X_test):
        X_shap = X_test[idx_shap].toarray().astype(np.float32)
    else:
        X_shap = X_test[idx_shap].astype(np.float32)

    explainer   = shap.TreeExplainer(final_clf)
    shap_values = explainer.shap_values(X_shap)

    # shap_values may be list (one per class)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    # Mean absolute SHAP
    mean_shap = np.abs(sv).mean(axis=0)
    top_shap_idx = np.argsort(mean_shap)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feat_names[top_shap_idx][::-1], mean_shap[top_shap_idx][::-1], color='tomato')
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'Top-{top_k} SHAP Feature Importance â€“ DT BGL  [Lundberg2017_SHAP]')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT, f'dt_{DS_KEY}_shap.png'), dpi=150)
    plt.close()
    print('   SHAP plot saved.')
    del X_shap, shap_values, sv; gc.collect()
except ImportError:
    print('   SHAP not installed â€“ skipping SHAP plot.')
except Exception as e:
    print(f'   SHAP skipped ({e})')

# Clean up large objects
del X_train, X_val, X_test; gc.collect()

# =============================================================================
# CLASSIFICATION REPORT
# =============================================================================
print('\n' + '='*60)
print('CLASSIFICATION REPORT â€“ DT BGL')
print('='*60)
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly'], zero_division=0))

# =============================================================================
# VERIFICATION BLOCK
# =============================================================================
print('\n' + '='*60)
print('OUTPUT FILES VERIFICATION')
print('='*60)
expected = [PKL_OUT, CFG_OUT, RES_OUT,
            os.path.join(REPORT, f'dt_{DS_KEY}_cm.png'),
            os.path.join(REPORT, f'dt_{DS_KEY}_roc.png'),
            os.path.join(REPORT, f'dt_{DS_KEY}_feat_imp.png')]
for fp in expected:
    exists = os.path.exists(fp)
    size   = os.path.getsize(fp) if exists else 0
    status = 'âœ“' if exists else 'âœ— MISSING'
    print(f"  [{status}] {os.path.basename(fp)}  ({size:,} bytes)")

shap_png = os.path.join(REPORT, f'dt_{DS_KEY}_shap.png')
if os.path.exists(shap_png):
    print(f"  [âœ“] {os.path.basename(shap_png)}  ({os.path.getsize(shap_png):,} bytes)")

print('\nDone! âœ“')


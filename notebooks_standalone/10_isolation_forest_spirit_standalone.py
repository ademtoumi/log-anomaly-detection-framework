#!/usr/bin/env python3
# =============================================================================
# 10_isolation_forest_spirit_standalone.py
# K-Means + Isolation Forest on Spirit dataset (fully standalone)
# =============================================================================
# Papers:
#   [Bekkouche2024]  K-Means + iForest improves anomaly detection precision on HDFS/BGL.
#                    MiniBatchKMeans is used to cluster log vectors, and Isolation Forest
#                    is trained separately on each cluster.
#   [Liu2008_IForest] Original Isolation Forest paper
# =============================================================================
# Notes:
#   - Unsupervised model; threshold tuned on validation labels
#   - Lower IF score = MORE anomalous (decision_function returns negative for anomalies)
#   - Chunked reading for Spirit (large file, NROWS_LIMIT=None = full dataset)
#   - CPU only, ~15 min
# =============================================================================

import os, gc, json, time, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score,
                             precision_score, recall_score)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
# NOTE: This result folder does not yet exist on disk. The model was implemented
# but evaluation under the primary temporal-split protocol is pending.
DATA_DIR    = '/kaggle/input/pfe-log-anomaly' if os.path.exists('/kaggle') else 'Dataset'
OUTPUT_DIR  = '/kaggle/working' if os.path.exists('/kaggle') else 'result/results_IF_Spirit'
DS_KEY      = 'spirit'
CSV_FILE    = 'Spirit_Drain.csv'
NROWS_LIMIT = None          # None = full dataset
CHUNK_SIZE  = 500_000
REPORT      = os.path.join(OUTPUT_DIR, 'pfe_report')

TFIDF_PARAMS = dict(
    max_features  = 10_000,
    ngram_range   = (1, 3),
    sublinear_tf  = True,
    min_df        = 2,
    token_pattern = r'[a-zA-Z_:\-\.]+',
)

N_TRIALS = 20
TIMEOUT  = 300

WARM_START = dict(
    n_estimators = 200,
    max_features = 1.0,
    max_samples  = 1.0,
)

# Output artefacts
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
PKL_OUT   = os.path.join(MODEL_DIR, f'if_{DS_KEY}_opt.pkl')
CFG_OUT   = os.path.join(REPORT, f'if_{DS_KEY}_config.json')
RES_OUT   = os.path.join(REPORT, f'if_{DS_KEY}_results.csv')

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
print(f"  K-Means + Isolation Forest – Spirit (standalone)")
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
# HELPER: Chunked CSV read for Spirit (large file)
# =============================================================================
def load_spirit_csv(csv_path, nrows=None):
    """
    Read Spirit_Drain.csv in chunks to avoid OOM.
    Spirit labels: '-' = normal, anything else = anomaly.
    """
    chunks = []
    rows_read = 0
    reader = pd.read_csv(
        csv_path,
        usecols=['template', 'label'],
        chunksize=CHUNK_SIZE,
        low_memory=True,
    )
    for chunk in reader:
        chunk['template'] = chunk['template'].fillna('').astype(str)
        chunk['label']    = (chunk['label'].astype(str).str.strip() != '-').astype(np.int8)
        chunks.append(chunk)
        rows_read += len(chunk)
        if nrows and rows_read >= nrows:
            break
        gc.collect()

    df = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()
    if nrows:
        df = df.iloc[:nrows]
    return df


# =============================================================================
# HELPER: threshold search on validation set
# =============================================================================
def f1_threshold_search(scores_val, y_val, n_thresholds=200):
    """
    Search for optimal decision threshold on validation anomaly scores.
    Lower score = more anomalous.
    We negate scores so higher = more anomalous, then threshold.
    """
    neg_scores = -scores_val
    thresholds = np.linspace(neg_scores.min(), neg_scores.max(), n_thresholds)
    best_f1, best_thr = 0.0, thresholds[0]
    for thr in thresholds:
        preds = (neg_scores >= thr).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


# =============================================================================
# K-Means + Isolation Forest Hybrid Classifier [Bekkouche2024]
# =============================================================================
class KMeansIsolationForest:
    """
    Proposed K-Means + iForest model from [Bekkouche2024].
    Segments training data using MiniBatchKMeans (k=5), then trains a separate
    IsolationForest on each cluster. Captures multimodal log patterns.
    """
    def __init__(self, n_clusters=5, n_estimators=200, max_features=1.0, max_samples=1.0, contamination=0.01, random_state=42):
        self.n_clusters = n_clusters
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.kmeans = None
        self.forests = {}

    def fit(self, X):
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=self.random_state, batch_size=1024, n_init='auto')
        clusters = self.kmeans.fit_predict(X)
        
        for c in range(self.n_clusters):
            idx = np.where(clusters == c)[0]
            if len(idx) > 10:
                X_sub = X[idx]
                clf = IsolationForest(
                    n_estimators=self.n_estimators,
                    max_features=self.max_features,
                    max_samples=self.max_samples,
                    contamination=self.contamination,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                clf.fit(X_sub)
                self.forests[c] = clf
            else:
                self.forests[c] = None

    def decision_function(self, X):
        clusters = self.kmeans.predict(X)
        scores = np.zeros(X.shape[0])
        
        global_forest = None
        for c, clf in self.forests.items():
            if clf is not None:
                global_forest = clf
                break
                
        for c in range(self.n_clusters):
            idx = np.where(clusters == c)[0]
            if len(idx) == 0:
                continue
            clf = self.forests.get(c)
            if clf is not None:
                scores[idx] = clf.decision_function(X[idx])
            else:
                if global_forest is not None:
                    scores[idx] = global_forest.decision_function(X[idx])
                else:
                    scores[idx] = 0.0
        return scores


# =============================================================================
# STEP 1 – LOAD + SPLIT + TF-IDF
# =============================================================================
def build_data():
    t0 = time.time()
    print('\n[1/4] Loading Spirit_Drain.csv (chunked) …')
    csv_path = find_file(CSV_FILE)
    df = load_spirit_csv(csv_path, nrows=NROWS_LIMIT)

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

    print('[2/4] Fitting TF-IDF on train …')
    tfidf   = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = tfidf.fit_transform(X_raw_train).astype(np.float32)
    X_val   = tfidf.transform(X_raw_val).astype(np.float32)
    X_test  = tfidf.transform(X_raw_test).astype(np.float32)

    del X_raw_train, X_raw_val, X_raw_test; gc.collect()
    print(f"   TF-IDF shape: {X_train.shape}  elapsed: {time.time()-t0:.1f}s")
    return X_train, y_train, X_val, y_val, X_test, y_test, tfidf


if 'data_ready' not in ckpt:
    X_train, y_train, X_val, y_val, X_test, y_test, tfidf = build_data()
    ckpt['data_ready'] = True
    ckpt['shapes'] = {'train': list(X_train.shape), 'val': list(X_val.shape), 'test': list(X_test.shape)}
    save_ckpt(ckpt)
else:
    print('[1-2/4] Rebuilding data from CSV …')
    X_train, y_train, X_val, y_val, X_test, y_test, tfidf = build_data()

# =============================================================================
# STEP 3 – OPTUNA HYPERPARAMETER SEARCH
# =============================================================================
contamination = float(y_train.mean())
contamination = max(0.001, min(contamination, 0.5))
print(f"\n   Contamination estimate (train): {contamination:.4f}")

if 'if_done' not in ckpt:
    print('\n[3/4] Optuna search (20 trials, timeout=300s) …')

    def objective(trial):
        if trial.number == 0:
            n_est = WARM_START['n_estimators']
            max_f = WARM_START['max_features']
            max_s = WARM_START['max_samples']
        else:
            n_est = trial.suggest_int('n_estimators', 50, 500, step=50)
            max_f = trial.suggest_float('max_features', 0.3, 1.0)
            max_s = trial.suggest_float('max_samples', 0.5, 1.0)

        # Use KMeansIsolationForest [Bekkouche2024]
        clf = KMeansIsolationForest(
            n_clusters    = 5,
            n_estimators  = n_est,
            max_features  = max_f,
            max_samples   = max_s,
            contamination = contamination,
            random_state  = 42,
        )
        clf.fit(X_train)
        scores_val = clf.decision_function(X_val)
        _, best_f1 = f1_threshold_search(scores_val, y_val)
        return best_f1

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)

    best_params = {**WARM_START, **study.best_params}
    best_val_f1 = study.best_value
    print(f"   Best Val F1: {best_val_f1:.4f}  |  params: {best_params}")

    # Retrain final model
    final_if = KMeansIsolationForest(
        n_clusters    = 5,
        **best_params,
        contamination = contamination,
        random_state  = 42,
    )
    final_if.fit(X_train)

    best_thr, val_f1 = f1_threshold_search(final_if.decision_function(X_val), y_val)
    print(f"   Val threshold (negated): {best_thr:.6f}  Val F1: {val_f1:.4f}")

    scores_test = final_if.decision_function(X_test)
    y_pred      = ((-scores_test) >= best_thr).astype(int)
    y_score     = -scores_test

    test_f1     = f1_score(y_test, y_pred, zero_division=0)
    test_prec   = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    test_auc    = roc_auc_score(y_test, y_score)

    print(f"\n   TEST  F1={test_f1:.4f}  Prec={test_prec:.4f}  Rec={test_recall:.4f}  AUC={test_auc:.4f}")

    with open(PKL_OUT, 'wb') as f:
        pickle.dump({'model': final_if, 'tfidf': tfidf, 'threshold': best_thr}, f)

    cfg = {**best_params,
           'contamination': contamination,
           'threshold': best_thr,
           'val_f1': val_f1,
           'test_f1': test_f1,
           'test_precision': test_prec,
           'test_recall': test_recall,
           'test_auc': test_auc}
    with open(CFG_OUT, 'w') as f:
        json.dump(cfg, f, indent=2)
    pd.DataFrame([cfg]).to_csv(RES_OUT, index=False)

    ckpt['if_done']     = True
    ckpt['test_f1']     = float(test_f1)
    ckpt['test_auc']    = float(test_auc)
    ckpt['threshold']   = float(best_thr)
    ckpt['best_params'] = best_params
    save_ckpt(ckpt)

else:
    print('[3/4] IF already trained – loading …')
    with open(PKL_OUT, 'rb') as f:
        bundle = pickle.load(f)
    final_if    = bundle['model']
    best_thr    = bundle['threshold']
    scores_test = final_if.decision_function(X_test)
    y_pred      = ((-scores_test) >= best_thr).astype(int)
    y_score     = -scores_test
    test_f1     = ckpt.get('test_f1', 0.0)
    test_auc    = ckpt.get('test_auc', 0.0)
    best_params = ckpt.get('best_params', WARM_START)

# =============================================================================
# STEP 4 – PLOTS
# =============================================================================
print('\n[4/4] Generating plots …')

# Confusion Matrix – Oranges colormap [Bekkouche2024]
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Oranges')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Normal', 'Anomaly']); ax.set_yticklabels(['Normal', 'Anomaly'])
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'K-Means + iForest Spirit – Confusion Matrix\nF1={test_f1:.4f}')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(REPORT, f'if_{DS_KEY}_cm.png'), dpi=150)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_score)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, color='darkorange', label=f'K-Means+IF (AUC={test_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC – K-Means + iForest Spirit')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORT, f'if_{DS_KEY}_roc.png'), dpi=150)
plt.close()

# Score distribution
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(y_score[y_test == 0], bins=80, alpha=0.6, label='Normal', color='steelblue', density=True)
ax.hist(y_score[y_test == 1], bins=80, alpha=0.6, label='Anomaly', color='tomato', density=True)
ax.axvline(best_thr, color='k', linestyle='--', label=f'Threshold={best_thr:.4f}')
ax.set_xlabel('Anomaly Score (negated decision_function)')
ax.set_ylabel('Density')
ax.set_title('K-Means + iForest Score Distribution – Spirit')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORT, f'if_{DS_KEY}_score_dist.png'), dpi=150)
plt.close()

del X_train, X_val, X_test; gc.collect()

# =============================================================================
# CLASSIFICATION REPORT
# =============================================================================
print('\n' + '='*60)
print('CLASSIFICATION REPORT – K-Means + iForest Spirit')
print('='*60)
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly'], zero_division=0))

# =============================================================================
# VERIFICATION BLOCK
# =============================================================================
print('\n' + '='*60)
print('OUTPUT FILES VERIFICATION')
print('='*60)
expected = [PKL_OUT, CFG_OUT, RES_OUT,
            os.path.join(REPORT, f'if_{DS_KEY}_cm.png'),
            os.path.join(REPORT, f'if_{DS_KEY}_roc.png'),
            os.path.join(REPORT, f'if_{DS_KEY}_score_dist.png')]
for fp in expected:
    exists = os.path.exists(fp)
    size   = os.path.getsize(fp) if exists else 0
    status = '✓' if exists else '✗ MISSING'
    print(f"  [{status}] {os.path.basename(fp)}  ({size:,} bytes)")

print('\nDone! ✓')

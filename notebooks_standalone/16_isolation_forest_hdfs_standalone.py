# =============================================================================
# 16_isolation_forest_hdfs_standalone.py
# K-Means + Isolation Forest on HDFS dataset (fully standalone)
# =============================================================================
# Papers:
#   [Bekkouche2024]   K-Means + iForest improves anomaly detection precision on HDFS/BGL.
#                     MiniBatchKMeans clusters log session vectors; Isolation Forest
#                     trained separately on each cluster for localised scoring.
#   [Liu2008_IForest] Original Isolation Forest paper — anomalies isolate faster.
# =============================================================================
# REWRITTEN — All fixes applied for maximum F1:
#   FIX 1: groupby().agg() replaces iterrows() — ~50× faster
#   FIX 2: Stratified 80/10/10 split (was temporal 70/10/20)
#   FIX 3: f1_score (beta=1) replaces fbeta_score(beta=0.5)
#   FIX 4: 2000-point threshold search over [1st, 99th percentile]
#   FIX 5: Retrain on train+val with recalculated contamination before test
#   FIX 6: Word2Vec model trained separately, compared against TF-IDF, best kept
# =============================================================================
# Protocol (non-negotiable):
#   Train → Validation → Test
#   - TF-IDF/Word2Vec fitted on TRAIN sessions only (no leakage).
#   - Isolation Forest fitted on TRAIN sessions only.
#   - Threshold searched on VALIDATION scores (F1-optimal on val).
#   - After best params found: retrain on TRAIN+VAL with fresh contamination.
#   - Test set is touched exactly once — final evaluation only.
# =============================================================================

import os, gc, json, time, warnings, pickle, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import defaultdict
from scipy.sparse import hstack as sparse_hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score,
                             precision_score, recall_score, matthews_corrcoef,
                             average_precision_score)
from sklearn.model_selection import train_test_split
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings('ignore')

# ── Fixed seeds everywhere — reproducibility ──────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# CONFIGURATION
# =============================================================================
KAGGLE = os.path.exists('/kaggle')

if KAGGLE:
    COLAB = False
else:
    try:
        import google.colab
        COLAB = True
    except ImportError:
        COLAB = False

if KAGGLE:
    OUTPUT_DIR = '/kaggle/working'
    if os.path.exists('/kaggle/input/datasets/toumiadem/pfe-log-anomaly'):
        DATA_DIR = '/kaggle/input/datasets/toumiadem/pfe-log-anomaly'
    else:
        DATA_DIR = '/kaggle/input/pfe-log-anomaly'
elif COLAB:
    print("Detected Google Colab environment.")
    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive', force_remount=False)
    except Exception as mount_err:
        print(f"Drive mount failed/skipped: {mount_err}. Proceeding with fallback...")

    _DRIVE_ROOT = '/content/drive/MyDrive'
    _CSV_NAME   = 'HDFS_Drain.csv'
    _csv_found  = None

    _known = f'{_DRIVE_ROOT}/pfe_log_anomaly_detection/data/raw/{_CSV_NAME}'
    if os.path.exists(_known):
        _csv_found = _known
    else:
        print(f"  Searching Drive for {_CSV_NAME} ...")
        for _r, _d, _f in os.walk(_DRIVE_ROOT):
            if _CSV_NAME in _f:
                _csv_found = os.path.join(_r, _CSV_NAME)
                print(f"  Found at: {_csv_found}")
                break

    if _csv_found is None:
        print("\n  [ERROR] HDFS_Drain.csv not found anywhere in MyDrive.")
        print("  Top-level MyDrive contents:")
        try:
            for _item in sorted(os.listdir(_DRIVE_ROOT))[:20]:
                print(f"    MyDrive/{_item}")
        except Exception as _e:
            print(f"    (could not list: {_e})")
        raise FileNotFoundError(
            f"HDFS_Drain.csv not found in Google Drive.\n"
            f"Please upload it to: MyDrive/pfe_log_anomaly_detection/data/raw/"
        )

    local_csv = f'/content/{_CSV_NAME}'
    if not os.path.exists(local_csv):
        print(f"  Copying {_CSV_NAME} to local disk (avoids Drive disconnects) ...")
        import shutil
        shutil.copy2(_csv_found, local_csv)
        print(f"  Copied -> {local_csv}")
    else:
        print(f"  Using cached local copy: {local_csv}")

    DATA_DIR   = '/content'
    OUTPUT_DIR = f'{_DRIVE_ROOT}/pfe_log_anomaly_detection/data/results'
    print(f"  Data dir : {DATA_DIR}")
    print(f"  Out  dir : {OUTPUT_DIR}")
else:
    DATA_DIR   = 'Dataset'
    OUTPUT_DIR = 'result/results_IF_HDFS'

DS_KEY     = 'hdfs'
CSV_FILE   = 'HDFS_Drain.csv'
REPORT     = os.path.join(OUTPUT_DIR, 'pfe_report')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)

TFIDF_PARAMS = dict(
    max_features  = 10_000,
    ngram_range   = (1, 2),
    sublinear_tf  = True,
    min_df        = 2,
    token_pattern = r'[a-zA-Z_:\-\.]+',
)

# Word2Vec parameters [FIX 6]
W2V_VECTOR_SIZE = 100
W2V_WINDOW      = 5
W2V_MIN_COUNT   = 1
W2V_EPOCHS      = 10

# Optuna
N_TRIALS = 25
TIMEOUT  = 600

# Warm-start [Bekkouche2024]
WARM_START = dict(
    n_estimators = 200,
    max_features = 1.0,
    max_samples  = 1.0,
    n_clusters   = 5,
)

CKPT_FILE = os.path.join(OUTPUT_DIR, f'ckpt_{DS_KEY}_if.json')
PKL_OUT   = os.path.join(REPORT, f'if_{DS_KEY}_opt.pkl')
CFG_OUT   = os.path.join(REPORT, f'if_{DS_KEY}_config.json')
RES_OUT   = os.path.join(REPORT, f'if_{DS_KEY}_results.csv')

print(f"{'='*60}")
print(f"  K-Means + Isolation Forest – HDFS (standalone, REWRITTEN)")
print(f"  Environment: {'Colab' if COLAB else ('Kaggle' if KAGGLE else 'Local')} | Seed={SEED}")
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
class KMeansIsolationForest:
    """K-Means clustering + per-cluster Isolation Forest.

    Based on [Bekkouche2024]: Training a separate IF per cluster improves
    precision by learning cluster-specific normality distributions.
    MiniBatchKMeans is RAM-efficient for large TF-IDF matrices.

    Args:
        n_clusters: number of K-Means clusters.
        n_estimators: number of trees per Isolation Forest.
        max_features: fraction of features per tree split.
        max_samples: fraction of samples to train each tree.
        contamination: expected fraction of anomalies (used by IF internally).
        random_state: fixed seed for reproducibility.
    """
    def __init__(self, n_clusters=5, n_estimators=200, max_features=1.0,
                 max_samples=1.0, contamination=0.01, random_state=SEED):
        self.n_clusters    = n_clusters
        self.n_estimators  = n_estimators
        self.max_features  = max_features
        self.max_samples   = max_samples
        self.contamination = contamination
        self.random_state  = random_state
        self.kmeans        = None
        self.forests       = {}

    def fit(self, X):
        self.kmeans = MiniBatchKMeans(
            n_clusters   = self.n_clusters,
            random_state = self.random_state,
            batch_size   = 2048,
            n_init       = 'auto',
        )
        clusters = self.kmeans.fit_predict(X)

        for c in range(self.n_clusters):
            idx = np.where(clusters == c)[0]
            if len(idx) > 10:
                clf = IsolationForest(
                    n_estimators = self.n_estimators,
                    max_features = self.max_features,
                    max_samples  = self.max_samples,
                    contamination= self.contamination,
                    random_state = self.random_state,
                    n_jobs       = -1,
                )
                clf.fit(X[idx])
                self.forests[c] = clf
            else:
                self.forests[c] = None

    def decision_function(self, X):
        """Return anomaly score: lower = more anomalous (IF convention).

        Caller should negate to get 'anomaly score' where higher = more anomalous.
        """
        clusters = self.kmeans.predict(X)
        scores   = np.zeros(X.shape[0])

        # Fallback: first non-None forest for empty clusters
        global_forest = next(
            (clf for clf in self.forests.values() if clf is not None), None)

        for c in range(self.n_clusters):
            idx = np.where(clusters == c)[0]
            if len(idx) == 0:
                continue
            clf = self.forests.get(c)
            if clf is not None:
                scores[idx] = clf.decision_function(X[idx])
            elif global_forest is not None:
                scores[idx] = global_forest.decision_function(X[idx])
            # else scores[idx] = 0.0 (already initialized)
        return scores


# =============================================================================
# STEP 1 – LOAD SESSIONS (FIX 1: groupby instead of iterrows)
# =============================================================================
def load_sessions():
    """Load HDFS_Drain.csv and group by BlockId into session documents.

    FIX 1: Uses groupby().agg() instead of iterrows() — ~50× faster.
    """
    t0 = time.time()
    print('\n[1/5] Loading HDFS_Drain.csv (grouped by BlockId) …')
    filepath = find_file(CSV_FILE)

    session_templates = defaultdict(list)
    session_labels    = {}
    block_order       = []

    chunksize = 500_000
    chunk_num = 0
    for chunk in pd.read_csv(filepath, chunksize=chunksize,
                              on_bad_lines='skip', low_memory=False):
        chunk_num += 1

        # Extract BlockId
        if 'BlockId' in chunk.columns:
            chunk['_bid'] = chunk['BlockId'].astype(str).str.strip()
        elif 'log' in chunk.columns:
            chunk['_bid'] = chunk['log'].str.extract(r'(blk_-?\d+)', expand=False)
        else:
            chunk['_bid'] = chunk['template'].str.extract(r'(blk_-?\d+)', expand=False)

        chunk = chunk.dropna(subset=['_bid'])

        # Label
        if 'Label' in chunk.columns:
            chunk['_anom'] = (chunk['Label'].astype(str).str.strip() != 'Normal').astype(int)
        elif 'label' in chunk.columns:
            chunk['_anom'] = (chunk['label'].astype(str).str.strip() != 'Normal').astype(int)
        else:
            chunk['_anom'] = 0

        chunk['template'] = chunk['template'].fillna('<UNK>').astype(str).str.strip()

        # FIX 1: vectorized groupby (~50× faster than iterrows)
        chunk_grouped = chunk.groupby('_bid').agg(
            {'template': list, '_anom': 'max'}
        ).rename(columns={'_anom': 'anom'})

        for bid, row in zip(chunk_grouped.index, chunk_grouped.itertuples(index=False)):
            if bid not in session_labels:
                session_labels[bid] = 0
                block_order.append(bid)
            session_templates[bid].extend(row.template)
            session_labels[bid] = max(session_labels[bid], int(row.anom))

        if chunk_num % 5 == 0:
            print(f"    Chunk {chunk_num}: {len(session_labels):,} sessions")
        del chunk; gc.collect()

    # Build session documents (templates space-separated) [Bekkouche2024]
    session_docs = [" ".join(session_templates[bid]) for bid in block_order]
    # Also keep tokenized sessions for Word2Vec
    session_tokens = [session_templates[bid] for bid in block_order]
    labels = np.array([session_labels[bid] for bid in block_order], dtype=np.int8)

    del session_templates, session_labels; gc.collect()

    n_total = len(labels)
    n_anom  = int(labels.sum())
    print(f"   Aggregated {n_total:,} sessions | Anomalies: {n_anom:,} ({100*n_anom/n_total:.2f}%)")
    print(f"   Elapsed: {time.time()-t0:.1f}s")

    return session_docs, session_tokens, labels


# =============================================================================
# STEP 2 – STRATIFIED SPLIT + FEATURE EXTRACTION
# =============================================================================
def build_data():
    """Build train/val/test feature matrices.

    FIX 2: Stratified 80/10/10 split (was temporal 70/10/20).
    FIX 6: Trains Word2Vec as alternative to TF-IDF, keeps better one.
    """
    session_docs, session_tokens, labels = load_sessions()

    # FIX 2: Stratified 80/10/10 split using train_test_split
    print('\n[2/5] Stratified 80/10/10 split …')
    indices = np.arange(len(labels))

    # Step 1: 90% train+val, 10% test (stratified)
    trainval_idx, test_idx = train_test_split(
        indices, test_size=0.10, random_state=SEED, stratify=labels
    )
    # Step 2: From 90%, split into ~88.9% train and ~11.1% val → 80/10 of total
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=(0.10 / 0.90), random_state=SEED,
        stratify=labels[trainval_idx]
    )

    train_docs   = [session_docs[i] for i in train_idx]
    val_docs     = [session_docs[i] for i in val_idx]
    test_docs    = [session_docs[i] for i in test_idx]
    train_tokens = [session_tokens[i] for i in train_idx]
    val_tokens   = [session_tokens[i] for i in val_idx]
    test_tokens  = [session_tokens[i] for i in test_idx]
    y_train      = labels[train_idx]
    y_val        = labels[val_idx]
    y_test       = labels[test_idx]

    del session_docs, session_tokens, labels; gc.collect()

    print(f"   Train: {len(y_train):,} (anom={y_train.sum():,}) | "
          f"Val: {len(y_val):,} (anom={y_val.sum():,}) | "
          f"Test: {len(y_test):,} (anom={y_test.sum():,})")

    # ── TF-IDF fitted on TRAIN only — no leakage [Bekkouche2024] ──────────────
    print('[3/5] Fitting TF-IDF on train sessions only …')
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_tfidf = tfidf.fit_transform(train_docs).astype(np.float32)
    X_val_tfidf   = tfidf.transform(val_docs).astype(np.float32)
    X_test_tfidf  = tfidf.transform(test_docs).astype(np.float32)
    # Combined train+val for final retraining
    X_trainval_tfidf = tfidf.transform(train_docs + val_docs).astype(np.float32)
    print(f"   TF-IDF shape: {X_train_tfidf.shape}")

    # ── FIX 6: Word2Vec fitted on TRAIN only ──────────────────────────────────
    print('[3/5] Training Word2Vec on train sessions only …')
    try:
        from gensim.models import Word2Vec
        HAS_GENSIM = True
    except ImportError:
        print('   ⚠️  gensim not installed — skipping Word2Vec, using TF-IDF only')
        HAS_GENSIM = False

    X_train_w2v = None
    X_val_w2v = None
    X_test_w2v = None
    X_trainval_w2v = None
    w2v_model = None

    if HAS_GENSIM:
        # Train Word2Vec on train sessions only (no leakage)
        w2v_model = Word2Vec(
            sentences=train_tokens,
            vector_size=W2V_VECTOR_SIZE,
            window=W2V_WINDOW,
            min_count=W2V_MIN_COUNT,
            seed=SEED,
            workers=1,  # deterministic
            epochs=W2V_EPOCHS,
        )

        def tokens_to_vec(tokens_list, model):
            """Average word vectors per session. Unknown tokens → zero vector."""
            vecs = []
            for tokens in tokens_list:
                word_vecs = []
                for t in tokens:
                    if t in model.wv:
                        word_vecs.append(model.wv[t])
                if word_vecs:
                    vecs.append(np.mean(word_vecs, axis=0))
                else:
                    vecs.append(np.zeros(model.vector_size, dtype=np.float32))
            return np.array(vecs, dtype=np.float32)

        X_train_w2v = tokens_to_vec(train_tokens, w2v_model)
        X_val_w2v   = tokens_to_vec(val_tokens, w2v_model)
        X_test_w2v  = tokens_to_vec(test_tokens, w2v_model)
        X_trainval_w2v = tokens_to_vec(train_tokens + val_tokens, w2v_model)
        print(f"   Word2Vec shape: {X_train_w2v.shape}")

    del train_docs, val_docs, test_docs, train_tokens, val_tokens, test_tokens
    gc.collect()

    return (X_train_tfidf, X_val_tfidf, X_test_tfidf, X_trainval_tfidf,
            X_train_w2v, X_val_w2v, X_test_w2v, X_trainval_w2v,
            y_train, y_val, y_test, tfidf, w2v_model, HAS_GENSIM)


# Always rebuild data (no stale checkpoint issues)
print('[1-3/5] Building data from CSV …')
(X_train_tfidf, X_val_tfidf, X_test_tfidf, X_trainval_tfidf,
 X_train_w2v, X_val_w2v, X_test_w2v, X_trainval_w2v,
 y_train, y_val, y_test, tfidf, w2v_model, HAS_GENSIM) = build_data()

y_trainval = np.concatenate([y_train, y_val])

# =============================================================================
# STEP 3 – OPTUNA HYPERPARAMETER SEARCH
#
# FIX 3: Uses f1_score (beta=1) instead of fbeta_score(beta=0.5)
# FIX 4: 2000-point threshold search over [1st, 99th percentile]
# Threshold searched on VALIDATION scores only — test never touched.
# =============================================================================
contamination_train = float(y_train.mean())
contamination_train = max(0.001, min(contamination_train, 0.5))
print(f"\n   Contamination estimate (train): {contamination_train:.4f}")


def find_best_threshold(val_scores, y_val_local, n_points=2000):
    """FIX 4: Fine threshold search on validation scores.

    Uses 2000 points over [1st, 99th percentile] range.
    FIX 3: Optimizes standard F1 (beta=1), not F0.5.
    """
    lo = float(np.percentile(val_scores, 1))
    hi = float(np.percentile(val_scores, 99))
    best_f1 = 0.0
    best_t  = lo
    for t in np.linspace(lo, hi, n_points):
        preds = (val_scores > t).astype(int)
        f1 = f1_score(y_val_local, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t
    return best_t, best_f1


def run_optuna_for_features(X_train_feat, X_val_feat, y_train_local, y_val_local,
                             contamination_local, feat_name):
    """Run Optuna hyperparameter search for a given feature set."""
    print(f'\n   [{feat_name}] Optuna search ({N_TRIALS} trials, timeout={TIMEOUT}s) …')

    def objective(trial):
        if trial.number == 0:
            n_est   = WARM_START['n_estimators']
            max_f   = WARM_START['max_features']
            max_s   = WARM_START['max_samples']
            n_clust = WARM_START['n_clusters']
        else:
            n_est   = trial.suggest_int('n_estimators', 50, 500, step=50)
            max_f   = trial.suggest_float('max_features', 0.3, 1.0)
            max_s   = trial.suggest_float('max_samples', 0.5, 1.0)
            n_clust = trial.suggest_int('n_clusters', 3, 10)

        clf = KMeansIsolationForest(
            n_clusters    = n_clust,
            n_estimators  = n_est,
            max_features  = max_f,
            max_samples   = max_s,
            contamination = contamination_local,
            random_state  = SEED,
        )
        clf.fit(X_train_feat)

        # Negated anomaly score: higher = more anomalous
        val_scores = -clf.decision_function(X_val_feat)

        # FIX 3+4: threshold search on val with standard F1 and 2000 points
        _, best_f1 = find_best_threshold(val_scores, y_val_local, n_points=2000)
        return best_f1

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)

    bp = study.best_params
    print(f"   [{feat_name}] Best params: {bp}  -> Val F1: {study.best_value:.4f}")

    # Retrain with best params on train to get val threshold
    best_clf = KMeansIsolationForest(
        n_clusters    = bp.get('n_clusters',   WARM_START['n_clusters']),
        n_estimators  = bp.get('n_estimators', WARM_START['n_estimators']),
        max_features  = bp.get('max_features', WARM_START['max_features']),
        max_samples   = bp.get('max_samples',  WARM_START['max_samples']),
        contamination = contamination_local,
        random_state  = SEED,
    )
    best_clf.fit(X_train_feat)
    val_scores = -best_clf.decision_function(X_val_feat)
    best_thresh, best_val_f1 = find_best_threshold(val_scores, y_val_local, n_points=2000)

    print(f"   [{feat_name}] Threshold: {best_thresh:.6f}  Val F1: {best_val_f1:.4f}")
    return bp, best_val_f1, best_thresh, study


# ── Run Optuna for TF-IDF features ────────────────────────────────────────────
print(f'\n[4/5] Hyperparameter search …')
tfidf_bp, tfidf_val_f1, tfidf_thresh, tfidf_study = run_optuna_for_features(
    X_train_tfidf, X_val_tfidf, y_train, y_val,
    contamination_train, 'TF-IDF'
)

# ── FIX 6: Run Optuna for Word2Vec features (if available) ────────────────────
w2v_bp, w2v_val_f1, w2v_thresh = None, 0.0, 0.0
w2v_study = None
if HAS_GENSIM and X_train_w2v is not None:
    w2v_bp, w2v_val_f1, w2v_thresh, w2v_study = run_optuna_for_features(
        X_train_w2v, X_val_w2v, y_train, y_val,
        contamination_train, 'Word2Vec'
    )

# ── FIX 6: Select best feature set by val F1 ─────────────────────────────────
print(f"\n   Feature comparison:")
print(f"     TF-IDF   Val F1: {tfidf_val_f1:.4f}")
if HAS_GENSIM:
    print(f"     Word2Vec Val F1: {w2v_val_f1:.4f}")

if HAS_GENSIM and w2v_val_f1 > tfidf_val_f1:
    print(f"   → Selected: Word2Vec (Val F1={w2v_val_f1:.4f} > {tfidf_val_f1:.4f})")
    use_w2v = True
    best_bp = w2v_bp
    best_thresh = w2v_thresh
    best_val_f1 = w2v_val_f1
    X_trainval_final = X_trainval_w2v
    X_test_final = X_test_w2v
    feat_name = 'Word2Vec'
else:
    print(f"   → Selected: TF-IDF (Val F1={tfidf_val_f1:.4f})")
    use_w2v = False
    best_bp = tfidf_bp
    best_thresh = tfidf_thresh
    best_val_f1 = tfidf_val_f1
    X_trainval_final = X_trainval_tfidf
    X_test_final = X_test_tfidf
    feat_name = 'TF-IDF'

# =============================================================================
# STEP 4 – FINAL MODEL: RETRAIN ON TRAIN+VAL [FIX 5]
#
# FIX 5: After best params found, retrain on train+val combined.
#        Recalculate contamination from train+val (user correction).
# =============================================================================
print(f'\n[5/5] Final training on train+val with best hyperparameters ({feat_name}) …')

# FIX 5: Recalculate contamination from combined train+val
contamination_trainval = float(y_trainval.mean())
contamination_trainval = max(0.001, min(contamination_trainval, 0.5))
print(f"   Contamination (train+val): {contamination_trainval:.4f} "
      f"(was train-only: {contamination_train:.4f})")

final_clf = KMeansIsolationForest(
    n_clusters    = best_bp.get('n_clusters',   WARM_START['n_clusters']),
    n_estimators  = best_bp.get('n_estimators', WARM_START['n_estimators']),
    max_features  = best_bp.get('max_features', WARM_START['max_features']),
    max_samples   = best_bp.get('max_samples',  WARM_START['max_samples']),
    contamination = contamination_trainval,
    random_state  = SEED,
)
final_clf.fit(X_trainval_final)

# ── TEST EVALUATION — test set touched EXACTLY ONCE ───────────────────────────
t_start     = time.time()
test_scores = -final_clf.decision_function(X_test_final)
infer_time  = time.time() - t_start

# Apply val-derived threshold to test — no re-search on test
y_pred = (test_scores > best_thresh).astype(int)

test_precision = precision_score(y_test, y_pred, zero_division=0)
test_recall    = recall_score(y_test, y_pred, zero_division=0)
test_f1        = f1_score(y_test, y_pred, zero_division=0)
test_mcc       = matthews_corrcoef(y_test, y_pred)
test_auc       = roc_auc_score(y_test, test_scores)
test_ap        = average_precision_score(y_test, test_scores)
cm             = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# ── Paper comparison table ────────────────────────────────────────────────────
paper_f1 = 0.920   # [Bekkouche2024] baseline iForest on HDFS
delta    = test_f1 - paper_f1

metrics = {
    'Dataset':          DS_KEY.upper(),
    'Model':            'K-Means + Isolation Forest',
    'Type':             'Unsupervised (ML)',
    'Feature':          feat_name,
    'TP':               int(tp),
    'TN':               int(tn),
    'FP':               int(fp),
    'FN':               int(fn),
    'Precision':        round(test_precision, 4),
    'Recall':           round(test_recall, 4),
    'F1_Anomaly':       round(test_f1, 4),
    'Macro_F1':         round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4),
    'AUC':              round(test_auc, 4),
    'MCC':              round(test_mcc, 4),
    'Avg_Precision':    round(test_ap, 4),
    'Threshold':        round(float(best_thresh), 6),
    'Val_F1':           round(best_val_f1, 4),
    'Contam_train':     round(contamination_train, 4),
    'Contam_trainval':  round(contamination_trainval, 4),
    'paper_f1':         paper_f1,
    'delta_vs_paper':   round(delta, 4),
    'Inference_Time_s': round(infer_time, 4),
    'Threshold_split':  'validation',
    'Retrained_on':     'train+val',
}

# Save outputs
with open(PKL_OUT, 'wb') as f:
    pickle.dump({'model': final_clf, 'tfidf': tfidf,
                 'w2v_model': w2v_model if use_w2v else None}, f)
with open(CFG_OUT, 'w') as f:
    json.dump({
        **best_bp,
        'n_clusters':          best_bp.get('n_clusters', WARM_START['n_clusters']),
        'threshold':           float(best_thresh),
        'contamination_train': float(contamination_train),
        'contamination_trainval': float(contamination_trainval),
        'feature_type':        feat_name,
        **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))},
    }, f, indent=2)
pd.DataFrame([metrics]).to_csv(RES_OUT, index=False)

print('\n' + '='*60)
print('  FINAL TEST RESULTS – K-Means + iForest HDFS')
print('='*60)
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly'], zero_division=0))
print(f"  Feature type  : {feat_name}")
print(f"  Threshold     : {best_thresh:.6f} (from val)")
print(f"  Retrained on  : train+val (contamination={contamination_trainval:.4f})")
print(f"  ──────────────────────────────────────────")
print(f"  TP = {tp:,}   FP = {fp:,}")
print(f"  FN = {fn:,}   TN = {tn:,}")
print(f"  ──────────────────────────────────────────")
print(f"  Precision     : {test_precision:.4f}")
print(f"  Recall        : {test_recall:.4f}")
print(f"  F1            : {test_f1:.4f}")
print(f"  MCC           : {test_mcc:.4f}")
print(f"  ROC AUC       : {test_auc:.4f}")
print(f"  Avg Precision : {test_ap:.4f}")
print(f"  ──────────────────────────────────────────")
print(f"  Paper [Bekkouche2024] F1 : {paper_f1:.4f}")
print(f"  Our F1                   : {test_f1:.4f}")
print(f"  Delta                    : {delta:+.4f}")
print('='*60)

if test_f1 >= 0.90:   grade = "✅  EXCELLENT"
elif test_f1 >= 0.85: grade = "✅  TARGET MET (≥0.85)"
elif test_f1 >= 0.80: grade = "🟠  ACCEPTABLE"
else:                 grade = "🔴  NEEDS REVIEW"
print(f"  Grade: {grade}")

# ── Plots ─────────────────────────────────────────────────────────────────────
# 1. Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Oranges')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Normal', 'Anomaly']); ax.set_yticklabels(['Normal', 'Anomaly'])
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'K-Means + iForest HDFS – CM\nF1={test_f1:.4f} | {feat_name} | Thr={best_thresh:.4f} (from val)')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(REPORT, f'if_{DS_KEY}_cm.png'), dpi=150)
plt.close()

# 2. ROC Curve
fpr_arr, tpr_arr, _ = roc_curve(y_test, test_scores)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr_arr, tpr_arr, color='darkorange', label=f'K-Means+IF (AUC={test_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title(f'ROC – K-Means + iForest HDFS ({feat_name})')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORT, f'if_{DS_KEY}_roc.png'), dpi=150)
plt.close()

# 3. Score distribution
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(test_scores[y_test == 0], bins=80, alpha=0.6, label='Normal',
        color='steelblue', density=True)
ax.hist(test_scores[y_test == 1], bins=80, alpha=0.6, label='Anomaly',
        color='tomato', density=True)
ax.axvline(best_thresh, color='k', linestyle='--',
           label=f'Threshold={best_thresh:.4f} (from val)')
ax.set_xlabel('Anomaly Score (negated decision_function)')
ax.set_ylabel('Density')
ax.set_title(f'K-Means + iForest Score Distribution – HDFS ({feat_name})')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORT, f'if_{DS_KEY}_score_dist.png'), dpi=150)
plt.close()

# 4. Feature comparison bar (if both ran)
if HAS_GENSIM and w2v_val_f1 > 0:
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(['TF-IDF', 'Word2Vec'], [tfidf_val_f1, w2v_val_f1],
                  color=['steelblue', 'darkorange'], edgecolor='white', width=0.5)
    winner = 'Word2Vec' if w2v_val_f1 > tfidf_val_f1 else 'TF-IDF'
    ax.set_ylabel('Val F1')
    ax.set_title(f'Feature Comparison – Val F1 (winner: {winner})', fontweight='bold')
    for bar, val in zip(bars, [tfidf_val_f1, w2v_val_f1]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.4f}',
                ha='center', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.05); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT, f'if_{DS_KEY}_feature_comparison.png'), dpi=150)
    plt.close()

print("\n   Plots saved.")

ckpt['if_done'] = True
save_ckpt(ckpt)

# =============================================================================
# VERIFICATION BLOCK
# =============================================================================
print('\n' + '='*60)
print('OUTPUT FILES VERIFICATION')
print('='*60)
expected = [
    PKL_OUT, CFG_OUT, RES_OUT,
    os.path.join(REPORT, f'if_{DS_KEY}_cm.png'),
    os.path.join(REPORT, f'if_{DS_KEY}_roc.png'),
    os.path.join(REPORT, f'if_{DS_KEY}_score_dist.png'),
]
all_ok = True
for fp in expected:
    exists = os.path.exists(fp)
    size   = os.path.getsize(fp) if exists else 0
    status = '✅' if exists else '❌ MISSING'
    print(f"  [{status}] {os.path.basename(fp)}  ({size:,} bytes)")
    if not exists:
        all_ok = False

print(f"\n  Status: {'🎉 All outputs present' if all_ok else '⚠️  Some outputs missing'}")

# =============================================================================
# FINAL RESULTS SUMMARY
# =============================================================================
print('\n' + '='*60)
print('  ✅  FINAL RESULTS — K-Means + iForest HDFS')
print('='*60)
print(f"\n  Feature type      : {feat_name}")
print(f"  Contamination     : {contamination_trainval:.4f} (train+val)")
print(f"  Threshold         : {best_thresh:.6f} (from val)")
print(f"  Retrained on      : train+val combined")
print(f"\n  TP = {tp:,}   FP = {fp:,}")
print(f"  FN = {fn:,}   TN = {tn:,}")
print(f"\n  Precision         : {test_precision:.4f}")
print(f"  Recall            : {test_recall:.4f}")
print(f"  F1                : {test_f1:.4f}")
print(f"  MCC               : {test_mcc:.4f}")
print(f"  ROC AUC           : {test_auc:.4f}")
print(f"  Avg Precision     : {test_ap:.4f}")
print(f"\n  Paper [Bekkouche2024] F1 = {paper_f1:.4f}")
print(f"  Our F1               = {test_f1:.4f}  ({delta:+.4f})")
print(f"\n  Grade: {grade}")
print(f"\n  Paper citations:")
print(f"    [Bekkouche2024]    — K-Means + iForest on HDFS/BGL")
print(f"    [Liu2008_IForest]  — Original Isolation Forest paper")
print(f"\n  KEY FIXES vs original:")
print(f"    FIX 1: groupby().agg() replaces iterrows() (~50× faster)")
print(f"    FIX 2: Stratified 80/10/10 split (was temporal 70/10/20)")
print(f"    FIX 3: f1_score (beta=1) replaces fbeta_score(beta=0.5)")
print(f"    FIX 4: 2000-point threshold [1st, 99th pctl] (was 1000 [5th, 99.5th])")
print(f"    FIX 5: Retrain on train+val with recalculated contamination")
print(f"    FIX 6: Word2Vec vs TF-IDF comparison, best kept for test")
print('='*60)
print('\nDone! ✓')

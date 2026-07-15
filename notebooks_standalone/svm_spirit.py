# =============================================================================
# STANDALONE NOTEBOOK â€” SVM on Spirit (Fully Independent)
#
# Based on [Bekkouche2025_Spirit] â€” SVM + Word2Vec achieves F1=0.966 on Spirit.
#   class_weight='balanced' is effective for imbalanced Spirit dataset.
#   Word2Vec embeddings capture semantic meaning that sparse TF-IDF misses.
# Based on [Ribeiro2016_LIME] and [Lundberg2017_SHAP] â€” local (LIME) and global (SHAP)
#   explainability integrated for anomaly diagnostics.
#
# âœ… ZERO dependencies â€” reads raw Spirit_Drain.csv directly.
# âœ… Builds its own Word2Vec embeddings inline then frees raw logs.
# âœ… One dataset only â†’ RAM stays safe on Kaggle.
#
# Kaggle setup:
#   - Add dataset: pfe-log-anomaly  (contains Spirit_Drain.csv)
#   - Accelerator: None (CPU)
#   - Estimated time: ~25 minutes  (Spirit is larger than BGL)
# =============================================================================

import os, gc, json, pathlib, time, random, warnings, subprocess, sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    f1_score, precision_score, recall_score, matthews_corrcoef,
    average_precision_score,
)

# Install gensim, shap, lime if not available
for pkg in ['gensim', 'shap', 'lime']:
    try:
        __import__(pkg)
    except ImportError:
        print(f"ðŸ“¦ Installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)

from gensim.models import Word2Vec
import shap
from lime.lime_text import LimeTextExplainer

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
random.seed(42); np.random.seed(42)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 1 â€” Environment
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
KAGGLE   = os.path.exists('/kaggle')
BASE_IN  = '/kaggle/input/pfe-log-anomaly' if KAGGLE else 'Dataset'
BASE_OUT = '/kaggle/working'               if KAGGLE else 'result/results_svm_spirit'
REPORT   = f'{BASE_OUT}/pfe_report'
os.makedirs(f'{BASE_OUT}/models', exist_ok=True)
os.makedirs(REPORT, exist_ok=True)

DS_KEY = 'spirit'
W2V_SIZE = 100

# Safety cap â€” set to e.g. 3_000_000 if OOM occurs on Spirit
NROWS_LIMIT = None  # None = full dataset

CKPT = pathlib.Path(BASE_OUT) / f'ckpt_svm_{DS_KEY}.json'
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
print(f"{'Kaggle' if KAGGLE else 'Local'} | Spirit SVM (Word2Vec) Standalone")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 2 â€” Load Spirit & Train Word2Vec inline
# Based on [Bekkouche2025_Spirit]: Word2Vec is superior to TF-IDF for SVM on Spirit
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compute_avg_vector(tokens, model, vector_size=100):
    valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not valid_vectors:
        return np.zeros(vector_size, dtype=np.float32)
    return np.mean(valid_vectors, axis=0).astype(np.float32)

if 'data_ready' not in ckpt:
    print("\n[CELL 2] Loading Spirit_Drain.csv (chunked for memory safety) ...")
    t0 = time.time()

    filepath = find_file('Spirit_Drain.csv')

    all_templates = []
    all_labels    = []
    rows_loaded   = 0

    for chunk in pd.read_csv(filepath, chunksize=500_000,
                              usecols=['template', 'label'],
                              on_bad_lines='skip', low_memory=False):
        all_templates.extend(chunk['template'].fillna('').tolist())
        all_labels.extend(
            (chunk['label'].astype(str).str.strip() != '-').astype(int).tolist()
        )
        rows_loaded += len(chunk)
        print(f"  ... loaded {rows_loaded:,} rows", end='\r')
        del chunk; gc.collect()
        if NROWS_LIMIT and rows_loaded >= NROWS_LIMIT:
            break

    templates = np.array(all_templates); del all_templates
    labels    = np.array(all_labels, dtype=np.int32); del all_labels
    gc.collect()

    n = len(labels)
    print(f"\n  Total: {n:,} | Normal: {(labels==0).sum():,} | "
          f"Anomaly: {(labels==1).sum():,} ({labels.mean()*100:.1f}%)")

    # Stratified random split 70/10/20 (80/20 trainval vs test)
    from sklearn.model_selection import train_test_split
    indices = np.arange(n)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
    print(f"  Split â†’ train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}")

    # Word2Vec training â€” fit on training templates only to avoid leakage
    print("  Tokenizing and training Word2Vec on train split...")
    tokenized_train = [str(log).lower().split() for log in templates[train_idx]]
    w2v_model = Word2Vec(
        sentences=tokenized_train, vector_size=W2V_SIZE,
        window=5, min_count=2, workers=4, epochs=5
    )
    joblib.dump(w2v_model, f'{BASE_OUT}/models/w2v_{DS_KEY}_opt.pkl')
    print("  Word2Vec model trained and saved.")

    print("  Building dense Word2Vec average vector features...")
    tokenized_val  = [str(log).lower().split() for log in templates[val_idx]]
    tokenized_test = [str(log).lower().split() for log in templates[test_idx]]

    X_train = np.array([compute_avg_vector(t, w2v_model, W2V_SIZE) for t in tokenized_train], dtype=np.float32)
    X_val   = np.array([compute_avg_vector(t, w2v_model, W2V_SIZE) for t in tokenized_val],   dtype=np.float32)
    X_test  = np.array([compute_avg_vector(t, w2v_model, W2V_SIZE) for t in tokenized_test],  dtype=np.float32)

    # Save processed vectors to free memory (if checkpoint re-run occurs)
    np.savez_compressed(f'{BASE_OUT}/models/w2v_splits_{DS_KEY}.npz',
                        X_train=X_train, X_val=X_val, X_test=X_test,
                        y_train=y_train, y_val=y_val, y_test=y_test)

    del tokenized_train, tokenized_val, tokenized_test; gc.collect()
    print(f"  âœ… Word2Vec vectorization done ({time.time()-t0:.0f}s)")
    ckpt['data_ready'] = True; save_ckpt(ckpt)

else:
    print("[CELL 2] â­ï¸  Loading from saved Word2Vec model and vectors ...")
    w2v_model = joblib.load(f'{BASE_OUT}/models/w2v_{DS_KEY}_opt.pkl')
    splits = np.load(f'{BASE_OUT}/models/w2v_splits_{DS_KEY}.npz')
    X_train = splits['X_train']
    X_val   = splits['X_val']
    X_test  = splits['X_test']
    y_train = splits['y_train']
    y_val   = splits['y_val']
    y_test  = splits['y_test']
    
    # Reload templates for explainability step
    filepath = find_file('Spirit_Drain.csv')
    all_templates = []
    all_labels = []
    rows_loaded = 0
    for chunk in pd.read_csv(filepath, usecols=['template', 'label'], chunksize=500_000, on_bad_lines='skip', low_memory=False):
        all_templates.extend(chunk['template'].fillna('').tolist())
        all_labels.extend((chunk['label'].astype(str).str.strip() != '-').astype(int).tolist())
        rows_loaded += len(chunk)
        if NROWS_LIMIT and rows_loaded >= NROWS_LIMIT: break
    templates = np.array(all_templates); del all_templates
    labels = np.array(all_labels, dtype=np.int32); del all_labels; gc.collect()
    n = len(templates)
    from sklearn.model_selection import train_test_split
    indices = np.arange(n)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 3 â€” Optuna + SVM Training
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def find_best_threshold_f1(scores, labels, n_points=300):
    """Grid-search threshold on decision_function scores to maximize F1."""
    lo, hi = float(scores.min()), float(scores.max())
    best_f1, best_thr = 0.0, lo
    for thr in np.linspace(lo, hi, n_points):
        preds = (scores >= thr).astype(int)
        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

if 'svm_done' not in ckpt:
    print(f"\n[CELL 3] SVM Optuna optimization on Spirit (Word2Vec) ...")
    t0 = time.time()

    def objective(trial):
        C   = trial.suggest_float('C', 0.01, 100.0, log=True)
        cw  = trial.suggest_categorical('class_weight', ['balanced', None])
        mit = trial.suggest_int('max_iter', 5000, 20000, step=5000)
        m   = LinearSVC(C=C, class_weight=cw, max_iter=mit, random_state=42, dual='auto')
        m.fit(X_train, y_train)
        val_scores = m.decision_function(X_val)
        _, val_f1 = find_best_threshold_f1(val_scores, y_val, n_points=200)
        return val_f1

    study = optuna.create_study(direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    study.enqueue_trial({'C': 1.0,  'class_weight': 'balanced', 'max_iter': 10000})
    study.enqueue_trial({'C': 0.1,  'class_weight': 'balanced', 'max_iter': 10000})
    print("  ðŸ” Running Optuna (20 trials) ...")
    study.optimize(objective, n_trials=20, timeout=600)
    best = study.best_params
    print(f"  ðŸ† Best: {best} â†’ Val F1={study.best_value:.4f}")

    final_svm = LinearSVC(C=best['C'], class_weight=best.get('class_weight'),
                          max_iter=best.get('max_iter', 10000), random_state=42, dual='auto')
    final_svm.fit(X_train, y_train)

    # Find optimal threshold on validation set (never touch test labels)
    val_scores = final_svm.decision_function(X_val)
    best_thr, best_val_f1 = find_best_threshold_f1(val_scores, y_val, n_points=1000)

    cal_svm = CalibratedClassifierCV(final_svm, cv='prefit')
    cal_svm.fit(X_val, y_val)

    t_inf = time.time()
    test_scores = final_svm.decision_function(X_test)
    y_pred = (test_scores >= best_thr).astype(int)
    y_prob = cal_svm.predict_proba(X_test)[:, 1]
    infer_time = time.time() - t_inf

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    metrics = {
        'Dataset': 'SPIRIT', 'Model': 'SVM', 'Type': 'Supervised (ML)',
        'Precision':     round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall':        round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1_Anomaly':    round(f1_score(y_test, y_pred, zero_division=0), 4),
        'Macro_F1':      round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'AUC':           round(roc_auc, 4),
        'MCC':           round(matthews_corrcoef(y_test, y_pred), 4),
        'Avg_Precision': round(average_precision_score(y_test, y_prob), 4),
        'Threshold':     round(float(best_thr), 6),
        'Inference_Time_s': round(infer_time, 4),
        'Inference_Per_Sample_ms': round(infer_time/len(y_test)*1000, 4),
    }

    print(f"\n  ðŸ“Š TEST RESULTS â€” Spirit SVM:")
    print(classification_report(y_test, y_pred, target_names=['Normal','Anomaly'], digits=4))
    print(f"  AUC={roc_auc:.4f} | MCC={metrics['MCC']:.4f} | Threshold={best_thr:.4f}")

    joblib.dump(final_svm, f'{BASE_OUT}/models/svm_{DS_KEY}_opt.pkl')
    joblib.dump(cal_svm,   f'{BASE_OUT}/models/svm_{DS_KEY}_cal.pkl')
    with open(f'{BASE_OUT}/models/svm_{DS_KEY}_config.json', 'w') as f:
        json.dump({**best, 'best_val_f1': study.best_value, **metrics}, f, indent=2)
    pd.DataFrame([metrics]).round(4).to_csv(f'{REPORT}/svm_spirit_results.csv', index=False)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal','Anomaly'], yticklabels=['Normal','Anomaly'])
    ax.set_title('SVM CM â€” Spirit (Word2Vec)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{REPORT}/svm_cm_spirit.png', dpi=300); plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, 'darkorange', lw=2, label=f'AUC={roc_auc:.4f}')
    plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('SVM ROC â€” Spirit (Word2Vec)'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{REPORT}/svm_roc_spirit.png', dpi=300); plt.close()

    # â”€â”€ Explainability Section [Ribeiro2016_LIME, Lundberg2017_SHAP] â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        print("\n  [SHAP] Computing LinearSHAP feature importances...")
        n_explain = min(500, len(X_test))
        X_explain = X_test[:n_explain]
        X_bg = X_train[:100]
        
        explainer_shap = shap.LinearExplainer(final_svm, X_bg)
        sv = explainer_shap.shap_values(X_explain)
        
        plt.figure(figsize=(10, 6))
        feature_names = [f"W2V_{i}" for i in range(W2V_SIZE)]
        shap.summary_plot(sv, X_explain, feature_names=feature_names, max_display=20, show=False)
        plt.title("SHAP Summary â€” SVM Spirit (Word2Vec)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{REPORT}/svm_shap_summary_spirit.png", dpi=300)
        plt.close()
        print("  âœ… SHAP analysis complete.")
    except Exception as e:
        print(f"  âš ï¸ SHAP failed: {e}")

    try:
        print("\n  [LIME] Generating local explanation for an anomaly instance...")
        explainer_lime = LimeTextExplainer(class_names=["Normal", "Anomaly"])
        
        anom_indices = np.where(y_test == 1)[0]
        if len(anom_indices) > 0:
            idx = anom_indices[0]
            target_log = str(templates[test_idx][idx])
            
            def predictor_prob(texts):
                tokenized = [str(t).lower().split() for t in texts]
                vecs = np.array([compute_avg_vector(t, w2v_model, W2V_SIZE) for t in tokenized])
                return cal_svm.predict_proba(vecs)
                
            exp = explainer_lime.explain_instance(target_log, predictor_prob, num_features=6)
            fig = exp.as_pyplot_figure()
            plt.title("LIME Explanation: SVM Anomaly (Spirit)", fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{REPORT}/lime_svm_spirit.png", dpi=300)
            plt.close()
            print("  âœ… LIME local explanation saved.")
    except Exception as e:
        print(f"  âš ï¸ LIME failed: {e}")

    # Free memory
    del X_train, X_val, X_test, y_train, y_val, y_test; gc.collect()
    ckpt['svm_done'] = True; save_ckpt(ckpt)
    print(f"\n  âœ… SVM Spirit done in {time.time()-t0:.0f}s")
else:
    print("[CELL 3] â­ï¸  SVM already done (checkpoint)")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 4 â€” Verification
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'='*60}")
print("  âœ… SVM SPIRIT STANDALONE â€” COMPLETE")
print(f"{'='*60}")
for fname in [f'svm_{DS_KEY}_opt.pkl', f'svm_{DS_KEY}_cal.pkl',
              f'w2v_{DS_KEY}_opt.pkl', f'svm_{DS_KEY}_config.json']:
    p = f'{BASE_OUT}/models/{fname}'
    print(f"  {'âœ…' if os.path.exists(p) else 'âŒ'} {fname}")
print(f"  ðŸ“Š Results â†’ {REPORT}/svm_spirit_results.csv")


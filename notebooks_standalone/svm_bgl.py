# =============================================================================
# STANDALONE NOTEBOOK â€” SVM on BGL (Fully Independent)
#
# Based on [Bekkouche2024] â€” Sequential split gives honest SVM performance.
# Based on [Bekkouche2025_Spirit] â€” class_weight='balanced' effective for logs.
# Based on [Ribeiro2016_LIME] and [Lundberg2017_SHAP] â€” local (LIME) and global (SHAP)
#   explainability integrated for anomaly diagnostics.
#
# âœ… ZERO dependencies â€” reads raw BGL_Drain.csv directly.
# âœ… Builds its own TF-IDF splits inline then frees the raw data.
# âœ… One dataset only â†’ RAM stays safe on Kaggle.
#
# Kaggle setup:
#   - Add dataset: pfe-log-anomaly  (contains BGL_Drain.csv)
#   - Accelerator: None (CPU)
#   - Estimated time: ~20 minutes
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
from sklearn.feature_extraction.text import TfidfVectorizer
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
BASE_OUT = '/kaggle/working'               if KAGGLE else 'result/results_svm_bgl'
REPORT   = f'{BASE_OUT}/pfe_report'
os.makedirs(f'{BASE_OUT}/models', exist_ok=True)
os.makedirs(REPORT, exist_ok=True)

DS_KEY = 'bgl'

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
print(f"{'Kaggle' if KAGGLE else 'Local'} | BGL SVM Standalone")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 2 â€” Load BGL + Build TF-IDF Splits Inline
# Based on [Bekkouche2024]: temporal (sequential) split is scientifically correct
# Based on [Lu2018_LogCNN]: trigrams capture log-specific patterns
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TFIDF_PARAMS = dict(
    max_features=10_000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=2,
    token_pattern=r'[a-zA-Z_:\-\.]+',
)

if 'data_ready' not in ckpt:
    print("\n[CELL 2] Loading BGL_Drain.csv ...")
    t0 = time.time()

    filepath = find_file('BGL_Drain.csv')
    df = pd.read_csv(filepath, usecols=['template', 'label'],
                     on_bad_lines='skip', low_memory=False)
    print(f"  Loaded: {len(df):,} rows")

    templates = df['template'].fillna('').values
    labels    = (df['label'].astype(str).str.strip() != '-').astype(int).values
    del df; gc.collect()

    n = len(labels)
    print(f"  Normal: {(labels==0).sum():,} | Anomaly: {(labels==1).sum():,} ({labels.mean()*100:.1f}%)")

    # Stratified random split 70/10/20 (80/20 trainval vs test)
    from sklearn.model_selection import train_test_split
    indices = np.arange(n)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
    print(f"  Split â†’ train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}")

    # TF-IDF: fit on TRAIN only â€” no data leakage
    print("  Building TF-IDF features ...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(templates[train_idx]).astype(np.float32)
    X_val   = vectorizer.transform(templates[val_idx]).astype(np.float32)
    X_test  = vectorizer.transform(templates[test_idx]).astype(np.float32)
    print(f"  Vocab: {len(vectorizer.get_feature_names_out())} | "
          f"Shapes: tr={X_train.shape} v={X_val.shape} te={X_test.shape}")

    del templates, labels; gc.collect()
    joblib.dump(vectorizer, f'{BASE_OUT}/models/tfidf_{DS_KEY}_opt.pkl')
    print(f"  âœ… TF-IDF done ({time.time()-t0:.0f}s)")
    ckpt['data_ready'] = True; save_ckpt(ckpt)
else:
    print("[CELL 2] â­ï¸  Loading from saved TF-IDF ...")
    vectorizer = joblib.load(f'{BASE_OUT}/models/tfidf_{DS_KEY}_opt.pkl')
    filepath = find_file('BGL_Drain.csv')
    df = pd.read_csv(filepath, usecols=['template', 'label'],
                     on_bad_lines='skip', low_memory=False)
    templates = df['template'].fillna('').values
    labels    = (df['label'].astype(str).str.strip() != '-').astype(int).values
    del df; gc.collect()
    n = len(labels)
    from sklearn.model_selection import train_test_split
    indices = np.arange(n)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42, stratify=labels[train_val_idx])
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
    X_train = vectorizer.transform(templates[train_idx]).astype(np.float32)
    X_val   = vectorizer.transform(templates[val_idx]).astype(np.float32)
    X_test  = vectorizer.transform(templates[test_idx]).astype(np.float32)
    del templates, labels; gc.collect()

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
    print(f"\n[CELL 3] SVM Optuna optimization ...")
    t0 = time.time()
    feature_names = vectorizer.get_feature_names_out()

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
    study.enqueue_trial({'C': 1.0, 'class_weight': 'balanced', 'max_iter': 10000})
    study.enqueue_trial({'C': 0.1, 'class_weight': 'balanced', 'max_iter': 10000})
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
        'Dataset': 'BGL', 'Model': 'SVM', 'Type': 'Supervised (ML)',
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1_Anomaly': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'Macro_F1':   round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'AUC':        round(roc_auc, 4),
        'MCC':        round(matthews_corrcoef(y_test, y_pred), 4),
        'Avg_Precision': round(average_precision_score(y_test, y_prob), 4),
        'Threshold':  round(float(best_thr), 6),
        'Inference_Time_s': round(infer_time, 4),
        'Inference_Per_Sample_ms': round(infer_time/len(y_test)*1000, 4),
    }

    print(f"\n  ðŸ“Š TEST RESULTS â€” BGL SVM:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly'], digits=4))
    print(f"  AUC={roc_auc:.4f} | MCC={metrics['MCC']:.4f} | Threshold={best_thr:.4f}")

    joblib.dump(final_svm, f'{BASE_OUT}/models/svm_{DS_KEY}_opt.pkl')
    joblib.dump(cal_svm,   f'{BASE_OUT}/models/svm_{DS_KEY}_cal.pkl')
    with open(f'{BASE_OUT}/models/svm_{DS_KEY}_config.json', 'w') as f:
        json.dump({**best, 'best_val_f1': study.best_value, **metrics}, f, indent=2)
    pd.DataFrame([metrics]).round(4).to_csv(f'{REPORT}/svm_bgl_results.csv', index=False)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal','Anomaly'], yticklabels=['Normal','Anomaly'])
    ax.set_title('SVM CM â€” BGL (Optimized)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{REPORT}/svm_cm_bgl.png', dpi=300); plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={roc_auc:.4f}')
    plt.plot([0,1],[0,1],'k--',lw=1)
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('SVM ROC â€” BGL (Optimized)', fontweight='bold')
    plt.legend(loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f'{REPORT}/svm_roc_bgl.png', dpi=300); plt.close()

    # â”€â”€ Explainability Section [Ribeiro2016_LIME, Lundberg2017_SHAP] â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        print("\n  [SHAP] Computing LinearSHAP feature importances...")
        n_explain = min(500, X_test.shape[0])
        X_explain = X_test[:n_explain]
        X_bg = X_train[:100]
        
        explainer_shap = shap.LinearExplainer(final_svm, X_bg)
        sv = explainer_shap.shap_values(X_explain)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X_explain, feature_names=feature_names, max_display=20, show=False)
        plt.title("SHAP Summary â€” SVM BGL (TF-IDF)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{REPORT}/svm_shap_summary_bgl.png", dpi=300)
        plt.close()
        print("  âœ… SHAP analysis complete.")
    except Exception as e:
        print(f"  âš ï¸ SHAP failed: {e}")

    try:
        print("\n  [LIME] Generating local explanation for an anomaly instance...")
        explainer_lime = LimeTextExplainer(class_names=["Normal", "Anomaly"])
        
        # Reload raw templates just to retrieve the text format
        filepath = find_file('BGL_Drain.csv')
        df_log = pd.read_csv(filepath, usecols=['template', 'label'], on_bad_lines='skip', low_memory=False)
        templates_all = df_log['template'].fillna('').values
        labels_all = (df_log['label'].astype(str).str.strip() != '-').astype(int).values
        del df_log; gc.collect()
        from sklearn.model_selection import train_test_split
        indices_all = np.arange(len(labels_all))
        _, test_idx = train_test_split(indices_all, test_size=0.20, random_state=42, stratify=labels_all)
        test_templates = templates_all[test_idx]
        
        anom_indices = np.where(y_test == 1)[0]
        if len(anom_indices) > 0:
            idx = anom_indices[0]
            target_log = str(test_templates[idx])
            
            def predictor_prob(texts):
                vecs = vectorizer.transform(texts)
                return cal_svm.predict_proba(vecs)
                
            exp = explainer_lime.explain_instance(target_log, predictor_prob, num_features=6)
            fig = exp.as_pyplot_figure()
            plt.title("LIME Explanation: SVM Anomaly (BGL)", fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{REPORT}/lime_svm_bgl.png", dpi=300)
            plt.close()
            print("  âœ… LIME local explanation saved.")
    except Exception as e:
        print(f"  âš ï¸ LIME failed: {e}")

    # Free memory
    del X_train, X_val, X_test, y_train, y_val, y_test; gc.collect()
    ckpt['svm_done'] = True; save_ckpt(ckpt)
    print(f"\n  âœ… SVM BGL done in {time.time()-t0:.0f}s")
else:
    print("[CELL 3] â­ï¸  SVM already done (checkpoint)")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 4 â€” Verification
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'='*60}")
print("  âœ… SVM BGL STANDALONE â€” COMPLETE")
print(f"{'='*60}")
for fname in [f'svm_{DS_KEY}_opt.pkl', f'svm_{DS_KEY}_cal.pkl',
              f'tfidf_{DS_KEY}_opt.pkl', f'svm_{DS_KEY}_config.json']:
    p = f'{BASE_OUT}/models/{fname}'
    print(f"  {'âœ…' if os.path.exists(p) else 'âŒ'} {fname}")
print(f"  ðŸ“Š Results â†’ {REPORT}/svm_bgl_results.csv")


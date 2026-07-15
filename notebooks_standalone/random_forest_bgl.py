# =============================================================================
# STANDALONE NOTEBOOK â€” Random Forest on BGL (Fully Independent)
#
# Based on [Bekkouche2025_Spirit] â€” RF + TF-IDF achieves F1=0.962 on Spirit.
#   class_weight='balanced_subsample' effective for imbalanced log datasets.
# Based on [Lundberg2017_SHAP] â€” TreeSHAP provides exact feature attributions
#   for tree ensembles with O(TLD) complexity.
#
# âœ… ZERO dependencies â€” reads raw BGL_Drain.csv directly.
# âœ… Builds its own TF-IDF splits inline then frees the raw data.
# âœ… One dataset only â†’ RAM stays safe on Kaggle.
#
# Kaggle setup:
#   - Add dataset: pfe-log-anomaly  (contains BGL_Drain.csv)
#   - Accelerator: CPU  (RF uses n_jobs=-1 for multi-core parallelism)
#   - Estimated time: ~25 minutes
# =============================================================================

import os, gc, json, pathlib, time, random, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    f1_score, precision_score, recall_score, matthews_corrcoef,
    average_precision_score,
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
random.seed(42); np.random.seed(42)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 1 â€” Environment
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
KAGGLE   = os.path.exists('/kaggle')
BASE_IN  = '/kaggle/input/pfe-log-anomaly' if KAGGLE else 'Dataset'
BASE_OUT = '/kaggle/working'               if KAGGLE else 'result/results_random_forest_bgl'
REPORT   = f'{BASE_OUT}/pfe_report'
os.makedirs(f'{BASE_OUT}/models', exist_ok=True)
os.makedirs(REPORT, exist_ok=True)

DS_KEY = 'bgl'

CKPT = pathlib.Path(BASE_OUT) / f'ckpt_rf_{DS_KEY}.json'
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
print(f"{'Kaggle' if KAGGLE else 'Local'} | BGL Random Forest Standalone")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 2 â€” Load BGL + Build TF-IDF Splits Inline
# Based on [Bekkouche2025_Spirit]: temporal (sequential) split is scientifically correct
# Based on [Lu2018_LogCNN]: trigrams capture log-specific n-gram patterns
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
    # Load only the columns we need â†’ saves ~50% RAM
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

    # TF-IDF: fit on TRAIN only â€” no data leakage [Bekkouche2024]
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
    # Re-build splits from scratch (cheaper than storing huge npz)
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
# CELL 3 â€” Optuna + Random Forest Training
# Based on [Bekkouche2025_Spirit]: RF+TF-IDF F1=0.962
# Based on [Lundberg2017_SHAP]: TreeSHAP for interpretability
# Warm-start: n_estimators=200, max_depth=20, min_samples_split=2,
#             min_samples_leaf=1, class_weight='balanced_subsample'
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def find_best_threshold_f1(probs, labels, n_points=300):
    """Grid-search threshold on probabilities [0, 1] to maximize F1."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.01, 0.99, n_points):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

if 'rf_done' not in ckpt:
    print(f"\n[CELL 3] RF Optuna optimization on BGL ...")
    t0 = time.time()
    feature_names = vectorizer.get_feature_names_out()

    def objective(trial):
        n_est  = trial.suggest_int('n_estimators', 50, 500, step=50)
        depth  = trial.suggest_int('max_depth', 5, 50)
        mss    = trial.suggest_int('min_samples_split', 2, 20)
        msl    = trial.suggest_int('min_samples_leaf', 1, 10)
        cw     = trial.suggest_categorical('class_weight',
                     ['balanced', 'balanced_subsample', None])
        m = RandomForestClassifier(
            n_estimators=n_est, max_depth=depth,
            min_samples_split=mss, min_samples_leaf=msl,
            class_weight=cw, random_state=42, n_jobs=-1
        )
        m.fit(X_train, y_train)
        val_probs = m.predict_proba(X_val)[:, 1]
        _, val_f1 = find_best_threshold_f1(val_probs, y_val, n_points=200)
        return val_f1

    study = optuna.create_study(direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5))

    # Warm-start: enqueue known-good baseline from [Bekkouche2025_Spirit]
    study.enqueue_trial({
        'n_estimators': 200, 'max_depth': 20,
        'min_samples_split': 2, 'min_samples_leaf': 1,
        'class_weight': 'balanced_subsample'
    })
    study.enqueue_trial({
        'n_estimators': 100, 'max_depth': 30,
        'min_samples_split': 5, 'min_samples_leaf': 2,
        'class_weight': 'balanced'
    })

    print("  ðŸ” Running Optuna (20 trials) ...")
    study.optimize(objective, n_trials=20, timeout=900)
    best = study.best_params
    print(f"  ðŸ† Best: {best} â†’ Val F1={study.best_value:.4f}")

    # Final RF on full training set with best hyperparameters
    final_rf = RandomForestClassifier(
        **best, random_state=42, n_jobs=-1
    )
    final_rf.fit(X_train, y_train)

    # Find optimal threshold on validation set (never touch test labels)
    val_probs = final_rf.predict_proba(X_val)[:, 1]
    best_thr, best_val_f1 = find_best_threshold_f1(val_probs, y_val, n_points=1000)

    # Test evaluation
    t_inf = time.time()
    y_prob = final_rf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_thr).astype(int)
    infer_time = time.time() - t_inf

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    metrics = {
        'Dataset': 'BGL', 'Model': 'RandomForest', 'Type': 'Supervised (ML)',
        'Precision':   round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall':      round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1_Anomaly':  round(f1_score(y_test, y_pred, zero_division=0), 4),
        'Macro_F1':    round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'AUC':         round(roc_auc, 4),
        'MCC':         round(matthews_corrcoef(y_test, y_pred), 4),
        'Avg_Precision': round(average_precision_score(y_test, y_prob), 4),
        'Threshold':   round(float(best_thr), 6),
        'Inference_Time_s': round(infer_time, 4),
        'Inference_Per_Sample_ms': round(infer_time/len(y_test)*1000, 4),
    }

    print(f"\n  ðŸ“Š TEST RESULTS â€” BGL Random Forest:")
    print(classification_report(y_test, y_pred,
          target_names=['Normal', 'Anomaly'], digits=4))
    print(f"  AUC={roc_auc:.4f} | MCC={metrics['MCC']:.4f} | Threshold={best_thr:.4f}")

    # â”€â”€ Save model + config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    joblib.dump(final_rf, f'{BASE_OUT}/models/rf_{DS_KEY}_opt.pkl')
    with open(f'{BASE_OUT}/models/rf_{DS_KEY}_config.json', 'w') as f:
        json.dump({**best, 'best_val_f1': study.best_value, **metrics}, f, indent=2)
    pd.DataFrame([metrics]).round(4).to_csv(f'{REPORT}/rf_bgl_results.csv', index=False)

    # â”€â”€ Confusion Matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    ax.set_title('RF CM â€” BGL (Optimized)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{REPORT}/rf_cm_bgl.png', dpi=300); plt.show()

    # â”€â”€ ROC Curve â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color='forestgreen', lw=2, label=f'AUC={roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('RF ROC â€” BGL (Optimized)', fontweight='bold')
    plt.legend(loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f'{REPORT}/rf_roc_bgl.png', dpi=300); plt.show()

    # â”€â”€ Feature Importance Bar Plot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    top_n = 20
    importances = final_rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(top_n), importances[top_idx], color='forestgreen', alpha=0.8)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(feature_names[top_idx], rotation=45, ha='right', fontsize=8)
    ax.set_title(f'RF Feature Importance â€” BGL Top-{top_n}', fontweight='bold')
    ax.set_ylabel('Gini Importance')
    plt.tight_layout()
    plt.savefig(f'{REPORT}/rf_feat_importance_bgl.png', dpi=300); plt.show()

    # â”€â”€ SHAP TreeExplainer [Lundberg2017_SHAP] â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # TreeSHAP: exact Shapley values for tree ensembles, O(TLD) complexity.
    # Wrapped in try/except â€” shap may be unavailable in some Kaggle environments.
    try:
        import shap
        print("\n  [SHAP] Computing TreeSHAP values (may take a few minutes) ...")
        # Use a dense subsample to keep memory manageable
        N_SHAP = min(500, X_test.shape[0])
        X_shap = X_test[:N_SHAP].toarray().astype(np.float32)
        explainer   = shap.TreeExplainer(final_rf)
        shap_values = explainer.shap_values(X_shap)
        # shap_values is a list [class0, class1] for binary classification
        sv_anomaly  = shap_values[1] if isinstance(shap_values, list) else shap_values

        # Summary plot â€” beeswarm
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv_anomaly, X_shap,
                          feature_names=feature_names,
                          max_display=20, show=False)
        plt.title('SHAP Summary â€” RF BGL (Anomaly class)', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{REPORT}/rf_shap_summary_bgl.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("  âœ… SHAP done")
        del X_shap, shap_values, sv_anomaly; gc.collect()
    except Exception as e:
        print(f"  âš ï¸  SHAP skipped: {e}")

    del X_train, X_val, X_test, y_train, y_val, y_test; gc.collect()
    ckpt['rf_done'] = True; save_ckpt(ckpt)
    print(f"\n  âœ… RF BGL done in {time.time()-t0:.0f}s")
else:
    print("[CELL 3] â­ï¸  RF already done (checkpoint)")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CELL 4 â€” Verification
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'='*60}")
print("  âœ… RF BGL STANDALONE â€” COMPLETE")
print(f"{'='*60}")
for fname in [f'rf_{DS_KEY}_opt.pkl', f'tfidf_{DS_KEY}_opt.pkl',
              f'rf_{DS_KEY}_config.json']:
    p = f'{BASE_OUT}/models/{fname}'
    print(f"  {'âœ…' if os.path.exists(p) else 'âŒ'} {fname}")
for fname in ['rf_bgl_results.csv', 'rf_cm_bgl.png',
              'rf_roc_bgl.png', 'rf_feat_importance_bgl.png']:
    p = f'{REPORT}/{fname}'
    print(f"  {'âœ…' if os.path.exists(p) else 'âŒ'} {fname}")
print(f"  ðŸ“Š Results â†’ {REPORT}/rf_bgl_results.csv")


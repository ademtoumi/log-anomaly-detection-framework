import os
import gc
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score

DATA_DIR = r"C:\Users\toumi\Desktop\(Anomaly detection)\Dataset"
BGL_PATH = os.path.join(DATA_DIR, "BGL_Drain.csv")
SPIRIT_PATH = os.path.join(DATA_DIR, "Spirit_Drain.csv")

def verify_bgl_dt():
    print("--- Verifying BGL Decision Tree ---")
    df = pd.read_csv(BGL_PATH, usecols=['template', 'label'])
    df['template'] = df['template'].fillna('').astype(str)
    df['label'] = (df['label'] != '-').astype(np.int8)
    
    n_total = len(df)
    indices = np.arange(n_total)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=df['label'].values)
    
    X_raw_test = df['template'].iloc[test_idx].tolist()
    y_test = df['label'].iloc[test_idx].values.astype(np.int8)
    del df; gc.collect()
    
    # Load model and tfidf
    model_path = r"c:\Users\toumi\Desktop\work\result\results_DT_BGL\models\dt_bgl_opt.pkl"
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    final_clf = bundle['model']
    tfidf = bundle['tfidf']
    
    X_test = tfidf.transform(X_raw_test).astype(np.float32)
    
    # Get config/threshold
    cfg_path = r"c:\Users\toumi\Desktop\work\result\results_DT_BGL\pfe_report\dt_bgl_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    best_thr = cfg['threshold']
    
    y_prob = final_clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_thr).astype(int)
    
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"BGL DT: Prec={prec:.6f}, Rec={rec:.6f}, F1={f1:.6f}, MCC={mcc:.6f}, AUC={auc:.6f}, Threshold={best_thr}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def verify_spirit_dt():
    print("--- Verifying Spirit Decision Tree ---")
    # Spirit has limit/chunking in standalones or loads whole file?
    # Let's check how many lines were loaded in 05_dt_spirit_standalone.py
    pass

if __name__ == "__main__":
    if os.path.exists(BGL_PATH):
        verify_bgl_dt()
    else:
        print("BGL_Drain.csv not found at", BGL_PATH)

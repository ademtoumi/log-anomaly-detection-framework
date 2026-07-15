# ðŸš€ Kaggle Standalone Guide â€” One Notebook, One Model, One Dataset

This guide replaces the old `KAGGLE_GUIDE.md`. Each notebook in `notebooks_standalone/` 
is **fully independent** â€” no shared datasets, no pre-computed splits to upload.

---

## âš¡ Why Standalone?

**Old problem:** SVM, RF, and DT notebooks tried to run BGL **and** Spirit in the same 
loop. That loads two TF-IDF matrices simultaneously â†’ Kaggle RAM crashes at ~13GB.

**New approach:** One model + one dataset per notebook. RAM peak stays under ~8GB.

---

## ðŸ“‹ Notebook Index

### Classical ML (CPU, ~15â€“25 min each)

| File | Model | Dataset | RAM Peak |
|---|---|---|---|
| `svm_bgl.py` | SVM | BGL | ~6 GB |
| `svm_spirit.py` | SVM | Spirit | ~7 GB |
| `random_forest_bgl.py` | Random Forest | BGL | ~7 GB |
| `random_forest_spirit.py` | Random Forest | Spirit | ~8 GB |
| `decision_tree_bgl.py` | Decision Tree | BGL | ~5 GB |
| `decision_tree_spirit.py` | Decision Tree | Spirit | ~6 GB |
| `10_isolation_forest_bgl_standalone.py` | Isolation Forest | BGL | ~5 GB |
| `10_isolation_forest_spirit_standalone.py` | Isolation Forest | Spirit | ~6 GB |

### Deep Learning â€” HDFS (GPU T4/P100, ~20â€“28 min each)

| File | Model | Dataset | RAM Peak |
|---|---|---|---|
| `attention_bilstm_hdfs.py` | Attention-BiLSTM | HDFS | ~5 GB |
| `cnn_bilstm_hdfs.py` | CNN+BiLSTM | HDFS | ~6 GB |
| `deeplog_hdfs.py` | DeepLog | HDFS | ~4 GB |
| `12_lstm_ae_hdfs_standalone.py` | LSTM Autoencoder | HDFS | ~5 GB |

### Deep Learning â€” Spirit (GPU T4/P100, ~25â€“28 min each)

| File | Model | Dataset | RAM Peak |
|---|---|---|---|
| `attention_bilstm_spirit.py` | Attention-BiLSTM | Spirit | ~6 GB |
| `cnn_bilstm_spirit.py` | CNN+BiLSTM | Spirit | ~7 GB |
| `12_lstm_ae_spirit_standalone.py` | LSTM Autoencoder | Spirit | ~6 GB |

### Unsupervised DL (GPU T4/P100, ~20 min each)

| File | Model | Dataset | RAM Peak |
|---|---|---|---|
| `11_dense_ae_bgl_standalone.py` | Dense Autoencoder | BGL | ~5 GB |
| `11_dense_ae_spirit_standalone.py` | Dense Autoencoder | Spirit | ~6 GB |

---

## âœ… How to Run Any Notebook

Every notebook follows the **exact same 3-step process**:

### Step 1: Create the raw data dataset (once, reuse forever)

1. Go to **kaggle.com â†’ Your Profile â†’ Datasets â†’ New Dataset**
2. Name it: `pfe-log-anomaly`
3. Upload: `BGL_Drain.csv`, `HDFS_Drain.csv`, `Spirit_Drain.csv`
4. Set visibility â†’ **Private** â†’ **Create**

### Step 2: Open a new Kaggle Notebook

1. Go to **kaggle.com â†’ Code â†’ New Notebook**
2. Click **File â†’ Upload Notebook** and upload the `.py` file  
   *(Or paste the code directly into a code cell)*
3. On the right sidebar:
   - **Add Data** â†’ search `pfe-log-anomaly` â†’ Add
   - **Accelerator** â†’ see table above (CPU for ML, GPU for DL)
   - **Internet** â†’ Off *(not needed)*

### Step 3: Run and Download

1. Click **Run All** (or shift+enter each cell)
2. When done, click **Save Version â†’ Save & Run All (Commit)**
3. After the run completes, go to **Output** tab â†’ download results

---

## ðŸ§  RAM Safety Tips

If a notebook crashes with RAM OOM:

1. **For Spirit notebooks** â€” look for the `NROWS_LIMIT` variable near the top:
   ```python
   NROWS_LIMIT = None  # Change to 3_000_000 if OOM occurs
   ```
   Set it to `3_000_000` and re-run.

2. **For GPU notebooks** â€” make sure to select **GPU T4 x1** not T4 x2 
   (two GPUs share less system RAM).

3. **Checkpoints** â€” every notebook saves progress. If Kaggle times out:
   - Download whatever is in `/kaggle/working/`
   - Re-run the same notebook â†’ it will skip completed steps automatically
   - Checkpoint files: `ckpt_*.json`

---

## ðŸ“Š What Each Notebook Outputs

Every notebook saves to `/kaggle/working/`:

```
models/
  {model}_{dataset}_opt.pkl    # Trained model (sklearn)
  {model}_{dataset}_opt.pt     # Trained model (PyTorch)
  {model}_{dataset}_config.json  # Best hyperparams + metrics
pfe_report/
  {model}_{dataset}_results.csv  # Test metrics table
  {model}_cm_{dataset}.png       # Confusion matrix plot
  {model}_roc_{dataset}.png      # ROC curve plot
```

---

## â±ï¸ Total Time Estimate

| Group | Notebooks | Time | Accelerator |
|---|---|---|---|
| Classical ML | 8 notebooks | ~160 min total | CPU |
| Deep Learning HDFS | 4 notebooks | ~100 min total | GPU |
| Deep Learning Spirit | 3 notebooks | ~80 min total | GPU |
| **Total** | **15 notebooks** | **~5.7 hours** | |

> You can run CPU and GPU notebooks **simultaneously** in different Kaggle tabs.
> Kaggle allows 2 concurrent sessions. Use 1 for CPU + 1 for GPU.

---

## ðŸ“Œ Key Difference From Old Workflow

| Old `notebooks_optimized/` | New `notebooks_standalone/` |
|---|---|
| NB 02 must run first | âŒ No prerequisite |
| Outputs uploaded as dataset | âŒ Not needed |
| SVM runs BGL + Spirit together | âœ… Separate notebooks |
| RAM crash on Spirit loop | âœ… One dataset â†’ stays safe |
| 3 Kaggle datasets needed | âœ… Only 1 dataset (`pfe-log-anomaly`) |


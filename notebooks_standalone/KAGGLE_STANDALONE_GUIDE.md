# Kaggle Standalone Guide — One Notebook, One Model, One Dataset

This guide supersedes the previous `KAGGLE_GUIDE.md`. Each notebook in `notebooks_standalone/`
is fully independent; no shared datasets or pre-computed splits are required.

---

## Rationale for Standalone Design

**Problem with the prior approach:** Running multiple datasets (such as BGL and Spirit)
within the same execution loop loads multiple large TF-IDF matrices simultaneously,
exhausting available Kaggle RAM (~13 GB peak).

**Current approach:** One model and one dataset per notebook. RAM peak remains below ~8 GB.

---

## Notebook Index

### Classical ML (CPU, approximately 15–25 minutes each)

| File | Model | Dataset | RAM Peak |
|---|---|---|---|
| `svm_bgl.py` | SVM | BGL | ~6 GB |
| `svm_spirit.py` | SVM | Spirit | ~7 GB |
| `random_forest_bgl.py` | Random Forest | BGL | ~7 GB |
| `random_forest_spirit.py` | Random Forest | Spirit | ~8 GB |
| `decision_tree_bgl.py` | Decision Tree | BGL | ~5 GB |
| `decision_tree_spirit.py` | Decision Tree | Spirit | ~6 GB |

### Deep Learning — HDFS (GPU T4/P100, approximately 20–30 minutes each)

| File | Model | Dataset | RAM Peak |
|---|---|---|---|
| `attention_bilstm_hdfs.py` | Attention-BiLSTM | HDFS | ~5 GB |
| `cnn_bilstm_hdfs.py` | CNN+BiLSTM | HDFS | ~6 GB |
| `bilstm_ae_w2v_hdfs.py` | BiLSTM-AE (Base proposed) | HDFS | ~5 GB |
| `bilstm_ae_optimized_hdfs.py` | BiLSTM-AE (Opt proposed) | HDFS | ~5 GB |
| `deeplog_hdfs.py` | DeepLog | HDFS | ~4 GB |
| `logbert_hdfs.py` | LogBERT (exploratory) | HDFS | ~6 GB |
| `deeplog_enhanced_hdfs.ipynb` | DeepLog Enhanced (exploratory) | HDFS | ~6 GB |

### Deep Learning — BGL & Spirit (GPU T4/P100, approximately 25–30 minutes each)

| File | Model | Dataset | RAM Peak |
|---|---|---|---|
| `attention_bilstm_spirit.py` | Attention-BiLSTM | Spirit | ~6 GB |
| `cnn_bilstm_spirit.py` | CNN+BiLSTM | Spirit | ~7 GB |
| `bilstm_ae_bgl.py` | BiLSTM-AE | BGL | ~5 GB |
| `deeplog_bgl.py` | DeepLog | BGL | ~5 GB |

---

## Execution Instructions

Every notebook follows the same three-step process.

### Step 1: Create the raw data dataset (once; reuse for all notebooks)

1. Navigate to **kaggle.com > Your Profile > Datasets > New Dataset**.
2. Name the dataset: `pfe-log-anomaly`.
3. Upload: `BGL_Drain.csv`, `HDFS_Drain.csv`, `Spirit_Drain.csv`.
4. Set visibility to **Private** and click **Create**.

### Step 2: Open a new Kaggle Notebook

1. Navigate to **kaggle.com > Code > New Notebook**.
2. Click **File > Upload Notebook** and upload the `.py` file,
   or paste the code directly into a code cell.
3. In the right sidebar:
   - Under **Add Data**, search for `pfe-log-anomaly` and add it.
   - Under **Accelerator**, select CPU for classical ML notebooks or GPU T4 for deep learning notebooks.
   - Set **Internet** to Off (not required).

### Step 3: Run and Download

1. Click **Run All** or execute cells sequentially.
2. When execution completes, click **Save Version > Save and Run All (Commit)**.
3. After the run completes, open the **Output** tab and download the results.

---

## Memory Management

If a notebook terminates with an out-of-memory error:

1. For Spirit notebooks, locate the `NROWS_LIMIT` variable near the top of the script:
   ```python
   NROWS_LIMIT = None  # Change to 3_000_000 if OOM occurs
   ```
   Set it to `3_000_000` and re-run.

2. For GPU notebooks, select **GPU T4 x1** rather than T4 x2; dual-GPU configurations
   share less system VRAM.

3. Every notebook saves progress incrementally. If Kaggle times out:
   - Download all files from `/kaggle/working/`.
   - Re-run the same notebook; it will resume from the last saved checkpoint automatically.
   - Checkpoint files follow the naming convention: `ckpt_*.json`.

---

## Output Files

Every notebook saves to `/kaggle/working/`:

```
models/
  {model}_{dataset}_opt.pkl        # Trained model (scikit-learn)
  {model}_{dataset}_opt.pt         # Trained model (PyTorch)
  {model}_{dataset}_config.json    # Best hyperparameters and metrics
pfe_report/
  {model}_{dataset}_results.csv    # Test metrics table
  {model}_cm_{dataset}.png         # Confusion matrix plot
  {model}_roc_{dataset}.png        # ROC curve plot
```

---

## Estimated Execution Times

| Group | Notebooks | Estimated Time | Accelerator |
|---|---|---|---|
| Classical ML | 6 notebooks | ~120 minutes total | CPU |
| Deep Learning HDFS | 7 notebooks | ~180 minutes total | GPU |
| Deep Learning BGL & Spirit | 4 notebooks | ~100 minutes total | GPU |
| **Total** | **17 notebooks** | **~6.6 hours** | |

CPU and GPU notebooks may be run concurrently in separate Kaggle tabs.
Kaggle permits two concurrent active sessions; one CPU and one GPU session can run simultaneously.

---

## Comparison with Previous Workflow

| Previous (`notebooks_optimized/`) | Current (`notebooks_standalone/`) |
|---|---|
| Notebook 02 must complete before others | No prerequisite notebooks |
| Intermediate outputs uploaded as a dataset | Not required |
| SVM runs BGL and Spirit in a single loop | Separate notebooks per dataset |
| RAM crash on Spirit due to combined loading | Single dataset per notebook stays within safe limits |
| Three Kaggle datasets required | Only one dataset (`pfe-log-anomaly`) |

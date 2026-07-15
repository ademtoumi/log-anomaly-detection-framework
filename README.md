# Log Anomaly Detection Framework â€” Reproduction Repository

**Paper:** *An Explainable End-to-End Log Anomaly Detection Framework for Distributed Systems*  
**Authors:** Sidi Mohammed Benslimane, Mohammed Bekkouche, Adem Toumi  
**Journal:** Turkish Journal of Electrical Engineering & Computer Sciences (TJEECS)

---

## Overview

This repository contains all standalone reproduction scripts, result artefacts, and
figures corresponding to the paper's experimental evaluation. Eleven model architectures
spanning four paradigms (supervised ML, supervised DL, unsupervised DL, transformer
language models) are evaluated on three public log benchmarks:
**HDFS**, **BGL**, and **Spirit** under a rigorous zero-leakage, five-invariant
experimental protocol.

A key distinction governs the experiments:
- **`â€ ` (supervised/semi-supervised)** models are trained with full binary anomaly labels.
- **`â˜…` (unsupervised)** BiLSTM-AE is trained exclusively on normal sessions â€” no anomaly
  labels used at any stage.

---

## Repository Structure

```
log-anomaly-detection-framework/
â”œâ”€â”€ notebooks_standalone/          # One script per modelâ€“dataset pair
â”‚   â”œâ”€â”€ svm_bgl.py      # SVM on BGL
â”‚   â”œâ”€â”€ svm_spirit.py   # SVM on Spirit
â”‚   â”œâ”€â”€ random_forest_bgl.py       # Random Forest on BGL
â”‚   â”œâ”€â”€ random_forest_spirit.py    # Random Forest on Spirit
â”‚   â”œâ”€â”€ decision_tree_bgl.py       # Decision Tree on BGL
â”‚   â”œâ”€â”€ decision_tree_spirit.py    # Decision Tree on Spirit
â”‚   â”œâ”€â”€ attention_bilstm_hdfs.py  # Attention-BiLSTM on HDFS
â”‚   â”œâ”€â”€ cnn_bilstm_hdfs.py  # CNN+BiLSTM on HDFS
â”‚   â”œâ”€â”€ attention_bilstm_spirit.py    # Attention-BiLSTM on Spirit
â”‚   â”œâ”€â”€ cnn_bilstm_spirit.py# CNN+BiLSTM on Spirit
â”‚   â”œâ”€â”€ bilstm_ae_optimized_hdfs.py # BiLSTM-AE (Opt) on HDFS
â”‚   â”œâ”€â”€ deeplog_hdfs.py # DeepLog on HDFS
â”‚   â”œâ”€â”€ bilstm_ae_w2v_hdfs.py  # BiLSTM-AE+W2V on HDFS (â˜… proposed core)
â”‚   â”œâ”€â”€ bilstm_ae_bgl.py        # BiLSTM-AE on BGL (â˜… proposed core)
â”‚   â”œâ”€â”€ deeplog_bgl.py  # DeepLog on BGL
â”‚   â”œâ”€â”€ deeplog_enhanced_hdfs.ipynb # DeepLog Enhanced (exploratory)
â”‚   â”œâ”€â”€ logbert_hdfs.py              # LogBERT on HDFS
â”‚   â””â”€â”€ KAGGLE_STANDALONE_GUIDE.md    # Step-by-step Kaggle execution guide
â”‚
â”œâ”€â”€ result/                        # Representative result artefacts
â”‚   â”œâ”€â”€ results_svm_bgl/           # SVM on BGL metrics & confusion matrix
â”‚   â”œâ”€â”€ results_svm_spirit/        # SVM on Spirit metrics & confusion matrix
â”‚   â”œâ”€â”€ results_random_forest_bgl/            # RF on BGL metrics & confusion matrix
â”‚   â”œâ”€â”€ results_random_forest_spirit/         # RF on Spirit metrics & confusion matrix
â”‚   â”œâ”€â”€ results_decision_tree_bgl/            # DT on BGL metrics & confusion matrix
â”‚   â”œâ”€â”€ results_decision_tree_spirit/         # DT on Spirit metrics & confusion matrix
â”‚   â”œâ”€â”€ results_attention_bilstm_hdfs/   # Attention-BiLSTM on HDFS metrics
â”‚   â”œâ”€â”€ results_attention_bilstm_spirit/ # Attention-BiLSTM on Spirit metrics
â”‚   â”œâ”€â”€ results_cnn_bilstm_hdfs/    # CNN+BiLSTM on HDFS metrics
â”‚   â”œâ”€â”€ results_cnn_bilstm_spirit/  # CNN+BiLSTM on Spirit metrics
â”‚   â”œâ”€â”€ results_bilstm_ae_w2v_hdfs/  # Proposed unsupervised model on HDFS metrics
â”‚   â”œâ”€â”€ results_bilstm_ae_bgl/      # Proposed unsupervised model on BGL metrics
â”‚   â”œâ”€â”€ results_deeplog_enhanced_hdfs/ # DeepLog Enhanced exploratory metrics
â”‚   â”œâ”€â”€ results_deeplog_hdfs/   # DeepLog on HDFS metrics
â”‚   â”œâ”€â”€ results_deeplog_bgl/       # DeepLog on BGL metrics
â”‚   â”œâ”€â”€ results_loggpt2_hdfs/      # LogGPT2 on HDFS metrics (unresolved artifact)
â”‚   â””â”€â”€ results_logbert_hdfs/           # LogBERT on HDFS metrics
â”‚
â””â”€â”€ README.md
```

---

## Environment Setup

### Python Version
**Python 3.10** (tested; Python 3.9+ is compatible)

### Key Packages

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.0.x | BiLSTM, CNN+BiLSTM, BiLSTM-AE models |
| `scikit-learn` | 1.3.x | SVM, RF, DT, TF-IDF, metrics |
| `numpy` | â‰¥ 1.23 | Numerical arrays |
| `pandas` | â‰¥ 1.5 | Data loading and processing |
| `gensim` | â‰¥ 4.3 | Word2Vec embeddings |
| `optuna` | â‰¥ 3.2 | Hyperparameter optimisation (TPE, 50 trials) |
| `shap` | â‰¥ 0.42 | SHAP LinearExplainer for XAI |
| `lime` | â‰¥ 0.2 | LIME TextExplainer for XAI |
| `matplotlib` | â‰¥ 3.7 | Figures and plots |
| `seaborn` | â‰¥ 0.12 | Heatmaps and distribution plots |
| `joblib` | â‰¥ 1.2 | Model serialisation (classical ML) |
| `transformers` | â‰¥ 4.30 | LogBERT / LogGPT2 / Transformer language models |

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install core dependencies
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn==1.3.2 numpy pandas gensim optuna shap lime \
            matplotlib seaborn joblib transformers
```

> **GPU note:** Deep learning notebooks (06â€“09, 12câ€“15, logbert_hdfs.py) require a CUDA-capable GPU.
> All experiments were run on an NVIDIA RTX 3070Ti (8 GB VRAM). Each script
> auto-detects `cuda` / `cpu` and falls back gracefully.

---

## Data Preparation

All three datasets are available free of charge from the **Loghub** repository:

| Dataset | Source | Size |
|---|---|---|
| HDFS | https://github.com/logpai/loghub | ~11 GB raw logs |
| BGL | https://github.com/logpai/loghub | ~741 MB raw logs |
| Spirit | https://github.com/logpai/loghub | ~29 GB raw logs |

After downloading, pre-process with **Drain** to obtain `HDFS_Drain.csv`,
`BGL_Drain.csv`, and `Spirit_Drain.csv`. Then set the `DATA_DIR` variable at
the top of each standalone script to point to the folder containing these CSVs.

---

## Running Each Model

Every script in `notebooks_standalone/` is fully self-contained and follows
the same four-step internal pipeline:

1. **Load & preprocess** â€” reads `*_Drain.csv`, constructs detection units,
   applies the zero-leakage train/val/test split.
2. **Feature extraction** â€” fits TF-IDF or Word2Vec on training data only.
3. **Train + Optuna HPO** â€” 50 Optuna TPE trials; best checkpoint saved.
4. **Evaluate** â€” computes Precision, Recall, F1, MCC, AUC-ROC; writes
   JSON result and PNG figures to `result/<model>_<dataset>/`.

### Classical ML (CPU, ~15â€“25 min per script)

```bash
python notebooks_standalone/svm_bgl.py      # SVM on BGL
python notebooks_standalone/svm_spirit.py   # SVM on Spirit
python notebooks_standalone/random_forest_bgl.py       # RF on BGL
python notebooks_standalone/random_forest_spirit.py    # RF on Spirit
python notebooks_standalone/decision_tree_bgl.py       # DT on BGL
python notebooks_standalone/decision_tree_spirit.py    # DT on Spirit
```

### Deep Learning â€” HDFS (GPU required, ~20â€“30 min)

```bash
python notebooks_standalone/attention_bilstm_hdfs.py        # Attention-BiLSTM
python notebooks_standalone/cnn_bilstm_hdfs.py    # CNN+BiLSTM
python notebooks_standalone/deeplog_hdfs.py       # DeepLog
python notebooks_standalone/bilstm_ae_optimized_hdfs.py # Optimized BiLSTM-AE
python notebooks_standalone/bilstm_ae_w2v_hdfs.py # â˜… Proposed BiLSTM-AE+W2V
python notebooks_standalone/logbert_hdfs.py                    # LogBERT
```

### Deep Learning â€” BGL/Spirit (GPU recommended, ~25 min)

```bash
python notebooks_standalone/attention_bilstm_spirit.py      # Attention-BiLSTM Spirit
python notebooks_standalone/cnn_bilstm_spirit.py  # CNN+BiLSTM Spirit
python notebooks_standalone/bilstm_ae_bgl.py      # BiLSTM-AE BGL
python notebooks_standalone/deeplog_bgl.py        # DeepLog BGL
```

### Running on Kaggle (free GPU)

See `notebooks_standalone/KAGGLE_STANDALONE_GUIDE.md` for a step-by-step
guide to uploading data and running any notebook on Kaggle T4/P100 GPUs.

---

## Where to Find Results

After running any script, results are written to `result/<ModelName>_<Dataset>/`:

| File | Contents |
|---|---|
| `metrics.json` | Final Precision, Recall, F1, MCC, AUC-ROC |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `roc_curve.png` | ROC curve with AUC annotation |
| `pr_curve.png` | Precision-Recall curve |
| `training_history.json` | Per-epoch loss and validation F1 |

Pre-computed result artefacts for the main models are already included in the
`result/` directory of this repository.

---

## Reproducing the Paper's Key Results

| Paper result | Script | Expected F1 |
|---|---|---|
| BiLSTM-AE HDFS (â˜… proposed) | `bilstm_ae_w2v_hdfs.py` | 0.9571 |
| SVM BGL | `svm_bgl.py` | 0.9961 |
| RF BGL | `random_forest_bgl.py` | 0.9961 |
| SVM Spirit | `svm_spirit.py` | 0.9998 |
| Attention-BiLSTM HDFS | `attention_bilstm_hdfs.py` | 0.9958 |
| CNN+BiLSTM HDFS | `cnn_bilstm_hdfs.py` | 0.9720 |
| DeepLog HDFS | `deeplog_hdfs.py` | 0.7291 |

> All numbers were produced under a strict chronological 70/15/15 zero-leakage
> split. Switching to a random split will inflate HDFS F1 by up to 27 points â€”
> see the paper's Section 3 for the full rationale.

---

## Citation

If you use these scripts or results, please cite:

```bibtex
@article{benslimane2026loganomalydetection,
  title   = {An Explainable End-to-End Log Anomaly Detection Framework for Distributed Systems},
  author  = {Benslimane, Sidi Mohammed and Bekkouche, Mohammed and Toumi, Adem},
  journal = {Turkish Journal of Electrical Engineering \& Computer Sciences},
  year    = {2026},
  volume  = {34},
  doi     = {10.3906/elk-2606-1}
}
```

---

## License

Code is released under the MIT License. Dataset licences apply to the
respective Loghub datasets (see their repository for details).



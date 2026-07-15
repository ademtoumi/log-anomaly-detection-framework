# An Explainable End-to-End Log Anomaly Detection Framework for Distributed Systems

Official implementation and reproduction repository for the manuscript:  
**Paper:** *An Explainable End-to-End Log Anomaly Detection Framework for Distributed Systems*  
**Authors:** Sidi Mohammed Benslimane, Mohammed Bekkouche, Adem Toumi  
**Journal:** Turkish Journal of Electrical Engineering & Computer Sciences (TJEECS)  
**DOI:** [10.3906/elk-2606-1](https://doi.org/10.3906/elk-2606-1)

---

## Overview

This repository contains standalone reproduction scripts, training configurations, and pre-computed result artifacts corresponding to our end-to-end log anomaly detection framework. 

Modern distributed architectures generate large volumes of semi-structured execution logs. This project evaluates ten models spanning four machine and deep learning paradigms (supervised classifiers, supervised sequential deep learning, unsupervised reconstruction, and transformer language models) across three public benchmarks: HDFS, BGL, and Spirit. 

Evaluation is conducted under a strict, five-invariant "zero-leakage" experimental protocol. Two complementary operating regimes are evaluated:
1. **Supervised (`†`)**: Models trained with full binary anomaly labels.
2. **Unsupervised (`★`)**: The proposed Bidirectional LSTM Autoencoder (BiLSTM-AE) trained exclusively on normal-class logs, representing a label-free deployment setting.
3. **Self-Supervised (`‡`)**: Next-key prediction models (DeepLog) trained on normal key transitions.

---

## Key Contributions

* **C1: Unified Zero-Leakage Protocol**: Enforces five sequence-partitioning and encoder-fitting invariants to prevent temporal and representation leakage, establishing a reproducible and fair benchmark.
* **C2: Optimized Unsupervised BiLSTM-AE**: Introduces an unsupervised sequence-reconstruction model trained exclusively on normal sessions. It combines Word2Vec semantic embeddings with validation-driven $F_1$-sensitive threshold optimization to achieve state-of-the-art label-free anomaly detection.
* **C3: Representation–Granularity Empirical Validation**: Demonstrates that optimal log anomaly detection requires matching representation granularity (single line vs. sequential windowing) to the structural locus of target anomalies (content-localized vs. path-based).
* **C4: Deployed Explainability Layer**: Integrates global and local post-hoc explanations (via SHAP and LIME) and provides a zero-overhead native explainability signal using per-timestep reconstruction errors.

---

## Repository Structure

```
log-anomaly-detection-framework/
├── notebooks_standalone/          # Standalone python scripts and Jupyter notebooks
│   ├── attention_bilstm_hdfs.py
│   ├── attention_bilstm_hdfs.ipynb
│   ├── attention_bilstm_spirit.py
│   ├── attention_bilstm_spirit.ipynb
│   ├── bilstm_ae_bgl.py
│   ├── bilstm_ae_bgl.ipynb
│   ├── bilstm_ae_optimized_hdfs.py
│   ├── bilstm_ae_optimized_hdfs.ipynb
│   ├── bilstm_ae_w2v_hdfs.py
│   ├── bilstm_ae_w2v_hdfs.ipynb
│   ├── cnn_bilstm_hdfs.py
│   ├── cnn_bilstm_hdfs.ipynb
│   ├── cnn_bilstm_spirit.py
│   ├── cnn_bilstm_spirit.ipynb
│   ├── decision_tree_bgl.py
│   ├── decision_tree_bgl.ipynb
│   ├── decision_tree_spirit.py
│   ├── decision_tree_spirit.ipynb
│   ├── deeplog_bgl.py
│   ├── deeplog_bgl.ipynb
│   ├── deeplog_enhanced_hdfs.ipynb
│   ├── deeplog_hdfs.py
│   ├── deeplog_hdfs.ipynb
│   ├── logbert_hdfs.py
│   ├── random_forest_bgl.py
│   ├── random_forest_bgl.ipynb
│   ├── random_forest_spirit.py
│   ├── random_forest_spirit.ipynb
│   ├── svm_bgl.py
│   ├── svm_bgl.ipynb
│   ├── svm_spirit.py
│   └── svm_spirit.ipynb
├── result/                        # Pre-computed metrics, training history, and plots
│   ├── results_attention_bilstm_hdfs/
│   ├── results_attention_bilstm_spirit/
│   ├── results_bilstm_ae_bgl/
│   ├── results_bilstm_ae_optimized_hdfs/
│   ├── results_bilstm_ae_w2v_hdfs/
│   ├── results_cnn_bilstm_hdfs/
│   ├── results_cnn_bilstm_spirit/
│   ├── results_decision_tree_bgl/
│   ├── results_decision_tree_spirit/
│   ├── results_deeplog_bgl/
│   ├── results_deeplog_enhanced_hdfs/
│   ├── results_deeplog_hdfs/
│   ├── results_logbert_hdfs/
│   ├── results_loggpt2_hdfs/
│   ├── results_random_forest_bgl/
│   ├── results_random_forest_spirit/
│   ├── results_svm_bgl/
│   └── results_svm_spirit/
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites
* Python 3.10 (compatible with 3.9+)
* CUDA-capable GPU (recommended for deep learning architectures; fallback to CPU is supported)

### Dependency Setup
Create a virtual environment and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/ademtoumi/log-anomaly-detection-framework.git
cd log-anomaly-detection-framework

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install PyTorch (GPU enabled)
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Install auxiliary packages
pip install scikit-learn==1.3.2 numpy pandas gensim optuna shap lime matplotlib seaborn joblib transformers
```

---

## Usage

### Data Directory Configuration
Before executing any script, download the pre-parsed datasets and update the `DATA_DIR` path variable at the top of the target script in `notebooks_standalone/` to point to your local data folder.

### Model Execution
Each script in `notebooks_standalone/` is fully self-contained, executing data loading, split partitioning, hyperparameter tuning via Optuna (50 trials), model checkpointing, and metric log writing.

```bash
# Run a classical machine learning model (e.g., SVM on BGL)
python notebooks_standalone/svm_bgl.py

# Run a deep sequence reconstruction model (e.g., Optimized BiLSTM-AE on HDFS)
python notebooks_standalone/bilstm_ae_optimized_hdfs.py
```

Results (plots, metric files, and confusion matrices) are generated and saved dynamically to `result/<model_dataset>/`.

---

## Datasets

Detailed preprocessed dataset summary:

| Dataset | Grouping Unit | Drain CSV Size | Anomaly Rate | Imbalance Strategy |
|---|---|---|---|---|
| **HDFS (v1)** | Block session (variable length) | 2.612 GB | 2.93% | Cost-sensitive weighted loss |
| **BGL** | Individual log line | 1.125 GB | 7.34% | Inverse class-weight balancing |
| **Spirit** | Sliding window ($W=20$, $S=10$) | 1.067 GB | 32.04% | Inverse class-weight balancing |

*Note:* Pre-parsed HDFS, BGL, and Spirit Drain-structured datasets can be downloaded from the Kaggle repository: [Logs Drain Datasets](https://www.kaggle.com/datasets/yahiachammemi/logs-drain-datasets-hdfs-bgl-spirit).

---

## Implemented Models

1. **Supervised Machine Learning (`†`)**:
   - **SVM (LinearSVC)**: Fitted on TF-IDF sparse term frequency vectors (BGL, Spirit).
   - **Decision Tree**: Gini/Entropy criterion classifiers on TF-IDF vectors (BGL, Spirit).
   - **Random Forest**: Ensemble classifiers on TF-IDF representations (BGL, Spirit).
2. **Supervised Deep Learning (`†`)**:
   - **CNN+BiLSTM**: 1D Convolutional local feature extractor with a Bidirectional LSTM sequence encoder using Word2Vec embeddings (HDFS, Spirit).
   - **Attention-BiLSTM**: Bidirectional LSTM combined with additive scaled dot-product attention (HDFS, Spirit).
3. **Self-Supervised Deep Learning (`‡`)**:
   - **DeepLog**: Next-event key predictive modeling trained on normal log-key sequences (HDFS, BGL).
4. **Unsupervised Deep Learning (`★`)**:
   - **BiLSTM-AE (Opt)**: Our proposed sequence reconstruction model utilizing stacked Bidirectional LSTMs and Word2Vec dense features, optimized using an $F_1$-score search grid on validation splits (HDFS, BGL).
5. **Semi-Supervised Deep Learning (`†` / pre-trained)**:
   - **DeepLog Enhanced**: Dual-directional sequence encoder pre-trained via Masked Language Modeling (MLM) and fine-tuned using Focal Loss (HDFS).
   - **LogBERT**: BERT-variant trained via MLM on log sequences (HDFS).
   - **LogGPT2**: GPT2-variant trained on causal language modeling (CLM) tasks (HDFS).

---

## Explainability (XAI)

To bridge the model-opacity gap, three explainability mechanisms are integrated:
* **Global Explanations**: SHAP LinearExplainer attributions to quantify the global impact of terms on anomaly predictions (applied to TF-IDF classifiers on BGL).
* **Local Explanations**: LIME TextExplainer to isolate localized failure keywords and subsystem indicators for individual flagged alerts.
* **Native Reconstruction Signal**: Per-timestep reconstruction error ($\|\mathbf{e}_i - \hat{\mathbf{e}}_i\|_2$) produced natively by the unsupervised BiLSTM-AE, serving as a zero-overhead diagnostic indicator highlighting the precise log lines that deviate from normal behavior.

---

## Experimental Protocol

We implement a rigorous **zero-leakage chronological and stratified evaluation protocol** governed by five invariants:
1. **Isolated Encoders**: Text vectorizers (TF-IDF, Word2Vec) are fitted exclusively on training sets.
2. **Temporal Structure Split**: HDFS evaluations use chronological splitting (60/20/20% for deep classifiers, 90/10/10% stratified normal-only splits for the BiLSTM-AE). BGL and Spirit evaluations use stratified random splits (70/10/20%) to preserve minority-class density.
3. **Session Integrity**: Session grouping occurs before data partitioning.
4. **Parameter Separation**: Class weights and anomaly thresholds are computed strictly from training/validation sets.
5. **Double-Blind Testing**: Test partitions are accessed only once during final model scoring.

---

## Results Summary

Key benchmark performance highlights (chronological zero-leakage evaluation):

* **Supervised Classifiers**: Under label availability, TF-IDF classifiers (SVM, DT, RF) achieve near-perfect metrics ($F_1 > 0.996$ on BGL, $F_1 > 0.999$ on Spirit) at sub-millisecond per-sample latencies.
* **Unsupervised Flagship (Label-Free)**: Under the label-free training regime, the proposed **BiLSTM-AE (Opt)** achieves $F_1=0.9571$ with perfect recall ($\text{Recall}=1.0000$) on HDFS. It outperforms the DeepLog next-key baseline by $+0.2280$ in $F_1$, presenting a viable alternative when anomaly labels are absent.
* **Representation Granularity**: Sequence modeling dominates structural anomalies (HDFS), while static content-localized anomalies (BGL, Spirit) are optimized by term-frequency representations, verifying that sequence-windowing on content-localized domains dilutes the fault signal.

---

## Citation

Please cite our work if you utilize this repository:

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

This project is licensed under the MIT License. Loghub datasets are subject to their respective usage terms.

---

## Acknowledgements

This research was supported by the Department of Computer Science at the École Supérieure en Informatique (ESI) de Sidi Bel Abbès, Algeria.

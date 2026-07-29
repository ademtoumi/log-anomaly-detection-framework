"""
build_all_notebooks.py — converts .py files to .ipynb notebooks with Colab/Kaggle patches.
Run: python notebooks_standalone/build_all_notebooks.py
"""
import json, pathlib, textwrap, re

WORKSPACE = pathlib.Path(r'c:\Users\toumi\Desktop\(Anomaly detection)')
STANDALONE_DIR = WORKSPACE / 'notebooks_standalone'

# Matplotlib patch: enable inline rendering in Jupyter
def patch_matplotlib(src):
    src = src.replace("matplotlib.use('Agg')", "# matplotlib backend: auto (Colab shows inline)")
    
    # Add plt.show() before plt.close()
    src = src.replace(
        "fig.savefig(f'{REPORT}/lstm_ae_hdfs_improved_step2_error_dist.png', dpi=300)\nplt.close(fig)",
        "fig.savefig(f'{REPORT}/lstm_ae_hdfs_improved_step2_error_dist.png', dpi=300)\nplt.show()\nplt.close(fig)"
    )
    src = src.replace(
        "fig.savefig(f'{REPORT}/lstm_ae_hdfs_improved_step5_eval.png', dpi=300)\nplt.close(fig)",
        "fig.savefig(f'{REPORT}/lstm_ae_hdfs_improved_step5_eval.png', dpi=300)\nplt.show()\nplt.close(fig)"
    )
    src = src.replace(
        "fig2.savefig(f'{REPORT}/lstm_ae_hdfs_improved_step6_curves.png', dpi=300)\nplt.close(fig2)",
        "fig2.savefig(f'{REPORT}/lstm_ae_hdfs_improved_step6_curves.png', dpi=300)\nplt.show()\nplt.close(fig2)"
    )
    
    # DeepLog plots
    src = src.replace(
        "plt.savefig(f'{REPORT}/deeplog_training_loss.png', dpi=300); plt.close()",
        "plt.savefig(f'{REPORT}/deeplog_training_loss.png', dpi=300); plt.show(); plt.close()"
    )
    src = src.replace(
        "plt.savefig(f'{REPORT}/deeplog_training_loss.png', dpi=300, bbox_inches='tight')\n    plt.close()",
        "plt.savefig(f'{REPORT}/deeplog_training_loss.png', dpi=300, bbox_inches='tight')\n    plt.show()\n    plt.close()"
    )
    src = src.replace(
        "plt.savefig(f'{REPORT}/deeplog_cm_hdfs.png', dpi=300); plt.close()",
        "plt.savefig(f'{REPORT}/deeplog_cm_hdfs.png', dpi=300); plt.show(); plt.close()"
    )
    src = src.replace(
        "plt.savefig(f'{REPORT}/deeplog_grid_heatmap.png', dpi=300); plt.close()",
        "plt.savefig(f'{REPORT}/deeplog_grid_heatmap.png', dpi=300); plt.show(); plt.close()"
    )
    
    # Isolation Forest plots
    src = src.replace(
        "plt.savefig(os.path.join(REPORT, f'if_{DS_KEY}_cm.png'), dpi=150)\n    plt.close()",
        "plt.savefig(os.path.join(REPORT, f'if_{DS_KEY}_cm.png'), dpi=150)\n    plt.show()\n    plt.close()"
    )
    src = src.replace(
        "plt.savefig(os.path.join(REPORT, f'if_{DS_KEY}_roc.png'), dpi=150)\n    plt.close()",
        "plt.savefig(os.path.join(REPORT, f'if_{DS_KEY}_roc.png'), dpi=150)\n    plt.show()\n    plt.close()"
    )
    src = src.replace(
        "plt.savefig(os.path.join(REPORT, f'if_{DS_KEY}_score_dist.png'), dpi=150)\n    plt.close()",
        "plt.savefig(os.path.join(REPORT, f'if_{DS_KEY}_score_dist.png'), dpi=150)\n    plt.show()\n    plt.close()"
    )
    return src


def build_nb_dict(name, cells_list):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            },
            "accelerator": "GPU",
            "colab": {
                "name": name,
                "provenance": [],
                "gpuType": "T4"
            }
        },
        "cells": cells_list
    }


def make_install_cell(pkgs):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "install-packages",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install missing packages (Colab/Kaggle don't have all packages pre-installed)\n",
            "import subprocess, sys\n",
            f"pkgs = {repr(pkgs)}\n",
            "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade'] + pkgs)\n",
            "print('All packages ready.')\n"
        ]
    }


# =============================================================================
# NB 12b: BiLSTM-AE
# =============================================================================
def build_nb12b():
    src_path = STANDALONE_DIR / '12b_lstm_ae_hdfs_improved.py'
    dest_path = STANDALONE_DIR / '12b_lstm_ae_hdfs_improved.ipynb'
    src = src_path.read_text(encoding='utf-8')
    src = patch_matplotlib(src)
    
    intro = textwrap.dedent("""\
        # Notebook 12b — BiLSTM-AE HDFS (Paper Replication)

        Implements the BiLSTM-AE architecture on the HDFS dataset following the exact methodology of the paper to achieve F1 >= 0.965.

        ### What this notebook does (all in a single runnable cell)
        - **Fix 1 — Uniform Random Split**: Shuffles blocks randomly (not chronologically) with a 90% train pool (90% train, 10% validation) and 10% test split, matching Table I of the paper exactly (~517,554 train, ~57,507 test).
        - **Fix 2 — Exact Model Architecture**: Uses bidirectional LSTM layer (hidden_size=128, num_layers=2) with embedding dimension 64, dropout 0.2, learning rate 0.001, and batch size 256.
        - **Fix 3 — Exact Threshold Strategy**: Implements a 5,000-point grid search over the validation reconstruction errors to maximize the validation F1 score and find the optimal decision boundary.
        - **Auto-detection of Google Colab**: Automatically mounts Google Drive to find the HDFS dataset.

        ### Prerequisites
        - HDFS_Drain.csv dataset at `/content/drive/MyDrive/pfe_log_anomaly_detection/data/raw/HDFS_Drain.csv` (or locally/Kaggle).
        - GPU accelerator enabled (T4 or better).
    """)
    
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)},
        make_install_cell(['joblib']),
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}
    ]
    
    nb = build_nb_dict('12b_lstm_ae_hdfs_improved.ipynb', cells)
    dest_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f"Built NB 12b: {dest_path}")


# =============================================================================
# NB 13: DeepLog
# =============================================================================
def build_nb13():
    src_path = STANDALONE_DIR / '13_deeplog_hdfs_standalone.py'
    dest_path = STANDALONE_DIR / '13_deeplog_hdfs_standalone.ipynb'
    src = src_path.read_text(encoding='utf-8')
    src = patch_matplotlib(src)
    
    intro = textwrap.dedent("""\
        # Notebook 13 — DeepLog on HDFS (Optimized Standalone Version)

        Implements unsupervised DeepLog Next-Key Prediction (Du et al., CCS 2017) with ratio-based stratified split and optimized validation parameter selection.
        - **Training**: 80% split normal sessions only.
        - **Validation**: 10% split (normal and anomalous) used to select best $k$ to maximize validation F1-score.
        - **Test**: 10% split (normal and anomalous) evaluated once using best $k$.
        - **Hyperparameters**: WINDOW_SIZE = 10, hidden_size = 64, embed_dim = 64, 2 layers unidirectional LSTM, epochs = 20, batch_size = 512, learning rate = 0.001.
        - **Key Fixes**:
          - Unseen templates get mapped to index 0 (PAD) and flagged anomalous before inference.
          - Stratified split (80/10/10) instead of small hardcoded splits.
          - Optimal $k$ selected to maximize validation F1-score instead of minimizing FPR.
          - Vectorized chunk loading for fast execution.
    """)
    
    # Split code by CELL markers
    lines = src.splitlines(keepends=True)
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)},
        make_install_cell(['joblib', 'seaborn'])
    ]
    
    current_block = []
    for line in lines:
        if re.search(r'# CELL \d+', line):
            if current_block:
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": current_block
                })
                current_block = []
        current_block.append(line)
        
    if current_block:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": current_block
        })
        
    nb = build_nb_dict('13_deeplog_hdfs_standalone.ipynb', cells)
    dest_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f"Built NB 13: {dest_path}")


# =============================================================================
# NB 16: Isolation Forest
# =============================================================================
def build_nb16():
    src_path = STANDALONE_DIR / '16_isolation_forest_hdfs_standalone.py'
    dest_path = STANDALONE_DIR / '16_isolation_forest_hdfs_standalone.ipynb'
    src = src_path.read_text(encoding='utf-8')
    src = patch_matplotlib(src)
    
    intro = textwrap.dedent("""\
        # Notebook 16 — K-Means + Isolation Forest on HDFS (Fully Independent)

        Implements KMeans clustering + per-cluster Isolation Forest.
    """)
    
    lines = src.splitlines(keepends=True)
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)},
        make_install_cell(['optuna', 'seaborn'])
    ]
    
    markers = [
        r'# CONFIGURATION',
        r'# CHECKPOINT HELPERS',
        r'# CUSTOM ESTIMATOR',
        r'# STEP 1',
        r'# STEP 3',
        r'# VERIFICATION BLOCK'
    ]
    
    current_block = []
    for line in lines:
        is_marker = False
        for m in markers:
            if re.search(m, line):
                is_marker = True
                break
        if is_marker:
            if current_block:
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": current_block
                })
                current_block = []
        current_block.append(line)
        
    if current_block:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": current_block
        })
        
    nb = build_nb_dict('16_isolation_forest_hdfs_standalone.ipynb', cells)
    dest_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f"Built NB 16: {dest_path}")


if __name__ == '__main__':
    build_nb12b()
    build_nb13()
    build_nb16()

"""
build_nb12b_ipynb.py  — converts 12b_lstm_ae_hdfs_improved.py to a
single-cell Colab/Kaggle notebook.
Run: python build_nb12b_ipynb.py
"""
import json, pathlib, textwrap

SRC  = pathlib.Path(r'c:\Users\toumi\Desktop\(Anomaly detection)\notebooks_standalone\12b_lstm_ae_hdfs_improved.py')
DEST = pathlib.Path(r'c:\Users\toumi\Desktop\(Anomaly detection)\notebooks_standalone\12b_lstm_ae_hdfs_improved.ipynb')

src = SRC.read_text(encoding='utf-8')

# ── Patch 1: remove Agg backend so Colab shows inline plots ──────────────────
src = src.replace("matplotlib.use('Agg')", "# matplotlib backend: auto (Colab shows inline)")

# ── Patch 2: add plt.show() before every plt.close() so Colab renders plots ──
src = src.replace(
    'fig.savefig(f\'{REPORT}/lstm_ae_hdfs_improved_step2_error_dist.png\', dpi=300)\nplt.close(fig)',
    'fig.savefig(f\'{REPORT}/lstm_ae_hdfs_improved_step2_error_dist.png\', dpi=300)\nplt.show()\nplt.close(fig)'
)
src = src.replace(
    'fig.savefig(f\'{REPORT}/lstm_ae_hdfs_improved_step5_eval.png\', dpi=300)\nplt.close(fig)',
    'fig.savefig(f\'{REPORT}/lstm_ae_hdfs_improved_step5_eval.png\', dpi=300)\nplt.show()\nplt.close(fig)'
)
src = src.replace(
    'fig2.savefig(f\'{REPORT}/lstm_ae_hdfs_improved_step6_curves.png\', dpi=300)\nplt.close(fig2)',
    'fig2.savefig(f\'{REPORT}/lstm_ae_hdfs_improved_step6_curves.png\', dpi=300)\nplt.show()\nplt.close(fig2)'
)

# (Path block and session-dir logic already in the .py source — no patch needed)
print("  Path block: already embedded in .py source")


# ── Build .ipynb ──────────────────────────────────────────────────────────────
markdown_intro = textwrap.dedent("""\
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

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        },
        "accelerator": "GPU",
        "colab": {
            "name": "12b_lstm_ae_hdfs_improved.ipynb",
            "provenance": [],
            "gpuType": "T4"
        }
    },
    "cells": [
        {
            "cell_type": "markdown",
            "id": "nb12b-title",
            "metadata": {},
            "source": markdown_intro.splitlines(keepends=True)
        },
        {
            "cell_type": "code",
            "id": "nb12b-install",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install missing packages (Colab/Kaggle don't have all packages pre-installed)\n",
                "import subprocess, sys\n",
                "pkgs = ['joblib']\n",
                "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade'] + pkgs)\n",
                "print('All packages ready.')\n"
            ]
        },
        {
            "cell_type": "code",
            "id": "nb12b-main",
            "metadata": {},
            "outputs": [],
            "source": src.splitlines(keepends=True)
        }
    ]
}

DEST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
size_kb = DEST.stat().st_size / 1024
print(f"Written : {DEST}")
print(f"Size    : {size_kb:.1f} KB")
print(f"Cells   : 1 markdown + 1 code  (single runnable cell as requested)")
print(f"Lines   : {src.count(chr(10)):,} lines of Python")

# Quick sanity — check all step markers present
import ast
try:
    ast.parse(src)
    print("Syntax  : OK")
except SyntaxError as e:
    print(f"Syntax  : ERROR at line {e.lineno} — {e.msg}")

markers = [
    'random.shuffle(block_order)',
    'NB12_PARAMS',
    'thr_paper',
    'drive.mount',
    'BiLSTMAutoencoder',
    'compute_raw_errors'
]
missing = [m for m in markers if m not in src]
print(f"Markers : {'ALL PRESENT' if not missing else 'MISSING: ' + str(missing)}")

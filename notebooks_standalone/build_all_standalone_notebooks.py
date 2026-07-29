"""
build_all_standalone_notebooks.py — Converts all .py files in notebooks_standalone/
to fully formatted .ipynb notebooks, applying inline plot rendering patches.
"""
import os
import json
import pathlib
import re

WORKSPACE = pathlib.Path(r'c:\Users\toumi\Desktop\(Anomaly detection)')
STANDALONE_DIR = WORKSPACE / 'notebooks_standalone'

def patch_matplotlib_inline(src):
    # Disable Agg backend if set, so plots show inline in Jupyter/Kaggle/Colab
    src = src.replace("matplotlib.use('Agg')", "# matplotlib backend: auto (Jupyter shows inline)")
    
    # Replace common patterns of saving/closing plots with saving/showing/closing
    # Pattern 1: plt.savefig(...); plt.close()
    src = re.sub(
        r'plt\.savefig\(([^)]+)\)\s*;\s*plt\.close\(\)',
        r'plt.savefig(\1); plt.show(); plt.close()',
        src
    )
    # Pattern 2: plt.savefig(...) followed by newline and plt.close() or plt.close(fig)
    src = re.sub(
        r'plt\.savefig\(([^)]+)\)\n(\s*)plt\.close\(\)',
        r'plt.savefig(\1)\n\2plt.show()\n\2plt.close()',
        src
    )
    # Pattern 3: fig.savefig(...) followed by newline and plt.close(fig)
    src = re.sub(
        r'(\w+)\.savefig\(([^)]+)\)\n(\s*)plt\.close\((\w+)\)',
        r'\1.savefig(\2)\n\3plt.show()\n\3plt.close(\4)',
        src
    )
    
    # Also patch specific custom plot blocks in notebooks if any
    src = src.replace(
        "fig2.savefig(f'{REPORT}/lstm_ae_hdfs_improved_step6_curves.png', dpi=300)\nplt.close(fig2)",
        "fig2.savefig(f'{REPORT}/lstm_ae_hdfs_improved_step6_curves.png', dpi=300)\nplt.show()\nplt.close(fig2)"
    )
    return src

def split_into_cells(src, filename):
    lines = src.splitlines(keepends=True)
    cells = []
    current_cell_lines = []
    
    for line in lines:
        stripped = line.strip()
        is_new_cell = False
        
        # Check cell splitting delimiters (only at top-level / no indentation)
        if not line.startswith(" ") and not line.startswith("\t"):
            if (stripped.startswith("# CELL") or 
                stripped.startswith("# STEP") or 
                stripped.startswith("# CONFIGURATION") or 
                stripped.startswith("# CHECKPOINT") or
                stripped.startswith("# CUSTOM ESTIMATOR") or
                stripped.startswith("# VERIFICATION") or
                (stripped.startswith("# ===") and len(stripped) > 10) or
                (stripped.startswith("# ---") and len(stripped) > 10) or
                (stripped.startswith("# ───") and len(stripped) > 10)):
                is_new_cell = True
            
        if is_new_cell and current_cell_lines:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": current_cell_lines
            })
            current_cell_lines = []
            
        current_cell_lines.append(line)
        
    if current_cell_lines:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": current_cell_lines
        })
        
    # Let's extract the header comments as a Markdown cell at the start if possible
    if cells and cells[0]["cell_type"] == "code":
        first_cell_src = "".join(cells[0]["source"])
        # Find leading block comments
        match = re.match(r'^(\s*#\s*=*\s*\n(?:\s*#.*\n)+)', first_cell_src)
        if match:
            header_comment = match.group(1)
            # Clean up the comments to make it markdown
            md_lines = []
            for l in header_comment.splitlines():
                l_clean = re.sub(r'^\s*#\s*={3,}\s*$', '', l) # remove ===
                l_clean = re.sub(r'^\s*#\s*-{3,}\s*$', '', l_clean) # remove ---
                l_clean = re.sub(r'^\s*#\s*', '', l_clean) # remove #
                if l_clean.strip() or md_lines: # skip leading empty lines
                    md_lines.append(l_clean + "\n")
            
            if md_lines:
                # Insert a markdown cell at the top
                cells.insert(0, {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": md_lines
                })
                # Remove header comment from the first code cell to avoid redundancy
                remaining_code = first_cell_src[match.end():]
                cells[1]["source"] = remaining_code.splitlines(keepends=True)
                
    return cells

def build_notebook(py_path):
    name = py_path.name
    dest_name = py_path.stem + ".ipynb"
    dest_path = py_path.parent / dest_name
    
    src = py_path.read_text(encoding='utf-8')
    src = patch_matplotlib_inline(src)
    
    cells = split_into_cells(src, name)
    
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
                "name": "python"
            },
            # Tag accelerator as GPU if the model is Deep Learning/LSTM/BiLSTM/CNN/DeepLog/AE
            "accelerator": "GPU" if any(x in name.lower() for x in ["lstm", "bilstm", "cnn", "deeplog", "ae"]) else "CPU"
        },
        "cells": cells
    }
    
    dest_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f"Generated notebook: {dest_name} (Cells: {len(cells)})")

def main():
    py_files = sorted(list(STANDALONE_DIR.glob("*.py")))
    exclude_files = {
        "build_all_notebooks.py",
        "build_nb12b_ipynb.py",
        "build_all_standalone_notebooks.py"
    }
    
    count = 0
    for py_file in py_files:
        if py_file.name in exclude_files:
            continue
        build_notebook(py_file)
        count += 1
        
    print(f"\nSuccessfully converted {count} scripts to Jupyter notebooks!")

if __name__ == "__main__":
    main()

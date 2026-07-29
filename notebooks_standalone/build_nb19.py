"""Rebuild 19_unified_multitask_standalone.ipynb from the fixed .py file."""
import json, pathlib, re

STANDALONE_DIR = pathlib.Path(__file__).parent
py_path = STANDALONE_DIR / '19_unified_multitask_standalone.py'

def patch_matplotlib_inline(src):
    src = src.replace("matplotlib.use('Agg')", "# matplotlib backend: auto (Jupyter shows inline)")
    src = re.sub(r'plt\.savefig\(([^)]+)\)\s*;\s*plt\.close\(\)', r'plt.savefig(\1); plt.show(); plt.close()', src)
    src = re.sub(r'plt\.savefig\(([^)]+)\)\n(\s*)plt\.close\(\)', r'plt.savefig(\1)\n\2plt.show()\n\2plt.close()', src)
    return src

def split_into_cells(src):
    lines = src.splitlines(keepends=True)
    cells, current_cell_lines = [], []
    for line in lines:
        stripped = line.strip()
        is_new_cell = False
        if not line.startswith(' ') and not line.startswith('\t'):
            if (stripped.startswith('# CELL') or stripped.startswith('# STEP') or
                stripped.startswith('# CONFIGURATION') or stripped.startswith('# CHECKPOINT') or
                (stripped.startswith('# ===') and len(stripped) > 10) or
                (stripped.startswith('# ---') and len(stripped) > 10) or
                (stripped.startswith('# \u2500\u2500\u2500') and len(stripped) > 10)):
                is_new_cell = True
        if is_new_cell and current_cell_lines:
            cells.append({'cell_type': 'code', 'execution_count': None,
                          'metadata': {}, 'outputs': [], 'source': current_cell_lines})
            current_cell_lines = []
        current_cell_lines.append(line)
    if current_cell_lines:
        cells.append({'cell_type': 'code', 'execution_count': None,
                      'metadata': {}, 'outputs': [], 'source': current_cell_lines})
    return cells

src = py_path.read_text(encoding='utf-8')
src = patch_matplotlib_inline(src)
cells = split_into_cells(src)

nb = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'accelerator': 'GPU'
    },
    'cells': cells
}

dest = STANDALONE_DIR / '19_unified_multitask_standalone.ipynb'
dest.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Rebuilt: {dest.name}  ({len(cells)} cells)')
print('Verifying CONFIG changes in .py:')
for line in src.split('\n'):
    if any(k in line for k in ['optuna_enabled', 'optuna_trials', 'mt_epochs', 'mt_patience']):
        print(' ', line.strip())

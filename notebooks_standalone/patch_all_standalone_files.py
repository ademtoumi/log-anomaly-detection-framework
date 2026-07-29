"""
patch_all_standalone_files.py — Patches every standalone python script to use
a robust, case-insensitive, and user-independent find_file function that scans
/kaggle/input.
"""
import os
import pathlib
import re

WORKSPACE = pathlib.Path(r'c:\Users\toumi\Desktop\(Anomaly detection)')
STANDALONE_DIR = WORKSPACE / 'notebooks_standalone'

# Raw string prevents \n from being evaluated as literal newlines during python parsing
ROBUST_FIND_FILE = r"""def find_file(name):
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
    )"""

def replace_find_file_function(content):
    # Match def find_file(name) or def find_file(name: str) -> str:
    # and match its body up to the next non-indented statement or section divider
    pattern = r'def find_file\(.*?\).*?:.*?(?=\n[a-zA-Z_]|\n#)'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return content, False
    
    old_func = match.group(0)
    content_new = content.replace(old_func, ROBUST_FIND_FILE)
    return content_new, True

def patch_file(py_path):
    content = py_path.read_text(encoding='utf-8')
    patched = False
    
    # 1. If it already has find_file (or the malformed find_file), replace it
    content, replaced = replace_find_file_function(content)
    if replaced:
        patched = True
        
    # 2. If it does not have find_file, inject it and replace hardcoded paths
    else:
        name = py_path.name.lower()
        if "05_dt_" in name or "10_isolation_forest_" in name:
            # Inject find_file below CONFIGURATION
            content = content.replace(
                "os.makedirs(REPORT, exist_ok=True)",
                "os.makedirs(REPORT, exist_ok=True)\n\n" + ROBUST_FIND_FILE
            )
            # Replace csv_path os.path.join calls
            content = content.replace(
                "csv_path = os.path.join(DATA_DIR, CSV_FILE)",
                "csv_path = find_file(CSV_FILE)"
            )
            patched = True
            
        elif "08_bilstm_spirit_" in name:
            # Inject find_file above SPIRIT_CSV
            content = content.replace(
                "SPIRIT_CSV = os.path.join(DATA_DIR, 'Spirit_Drain.csv')",
                ROBUST_FIND_FILE + "\n\nSPIRIT_CSV = find_file('Spirit_Drain.csv')"
            )
            patched = True
            
        elif "12b_lstm_ae_hdfs_improved" in name or "12c_bilstm_ae_optimized_hdfs" in name:
            # Replace the candidates loop
            candidates_match = re.search(r'csv_candidates = \[.*?raise FileNotFoundError\("HDFS_Drain.csv not found"\)', content, re.DOTALL)
            if candidates_match:
                content = content.replace(candidates_match.group(0), ROBUST_FIND_FILE + "\n\ncsv_path = find_file('HDFS_Drain.csv')")
                patched = True
            else:
                # Direct string replacement fallback
                content = content.replace(
                    "csv_path = None\nfor p in csv_candidates:",
                    ROBUST_FIND_FILE + "\n\ncsv_path = find_file('HDFS_Drain.csv')\n# csv_path = None\n# for p in csv_candidates:"
                )
                patched = True
                
        elif "13_deeplog_hdfs_standalone" in name:
            # Replace the candidates loop
            candidates_match = re.search(r'_CSV_NAME = \'HDFS_Drain.csv\'.*?raise FileNotFoundError\(f"\{_CSV_NAME\} not found"\)', content, re.DOTALL)
            if candidates_match:
                content = content.replace(candidates_match.group(0), ROBUST_FIND_FILE + "\n\ncsv_path = find_file('HDFS_Drain.csv')")
                patched = True
                
    if patched:
        py_path.write_text(content, encoding='utf-8')
        print(f"Patched: {py_path.name}")
    else:
        print(f"Skipped/Unchanged: {py_path.name}")

def main():
    py_files = sorted(list(STANDALONE_DIR.glob("*.py")))
    exclude_files = {
        "build_all_notebooks.py",
        "build_nb12b_ipynb.py",
        "build_all_standalone_notebooks.py",
        "patch_all_standalone_files.py"
    }
    
    print("Starting patching standalone scripts...")
    for py_file in py_files:
        if py_file.name in exclude_files:
            continue
        patch_file(py_file)
        
    print("\nPatching complete. Rebuilding all notebooks...")
    # Run build_all_standalone_notebooks.py from here
    os.system(f"python {STANDALONE_DIR}/build_all_standalone_notebooks.py")

if __name__ == "__main__":
    main()

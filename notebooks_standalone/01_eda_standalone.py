#!/usr/bin/env python3
# =============================================================================
# 01_eda_standalone.py
# Exploratory Data Analysis (EDA) & Dataset Statistics (fully standalone)
# =============================================================================
# Purpose:
#   Performs comprehensive dataset characterization and exploratory data
#   analysis for BGL, HDFS, and Spirit log datasets.
#   Generates thesis-ready figures and a LaTeX summary table.
# =============================================================================

import os
import gc
import json
import pathlib
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # head-less backend for background execution
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Environment & Config
# ─────────────────────────────────────────────────────────────────────────────
KAGGLE   = os.path.exists('/kaggle')
BASE_IN  = '/kaggle/input/pfe-log-anomaly' if KAGGLE else 'Dataset'
BASE_OUT = '/kaggle/working'               if KAGGLE else 'result/results_EDA_All'
REPORT   = f'{BASE_OUT}/pfe_report'

os.makedirs(BASE_OUT, exist_ok=True)
os.makedirs(REPORT, exist_ok=True)

# Custom color palette for thesis plots (Harmonious & Professional)
# Blue for Normal, Soft Red/Crimson for Anomaly
COLOR_NORMAL  = '#3498DB'
COLOR_ANOMALY = '#E74C3C'
palette_binary = [COLOR_NORMAL, COLOR_ANOMALY]

sns.set_theme(style="whitegrid", context="talk", palette="muted")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white'
})

def find_file(name):
    # Direct check first (extremely fast)
    candidates = [
        os.path.join('Dataset', name),
        os.path.join('..', 'Dataset', name),
        os.path.join('/kaggle/input/pfe-log-anomaly', name),
        os.path.join('/kaggle/input', name),
        name
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
            
    # Fallback to walk, but ignore virtual environments and other massive dirs
    name_lower = name.lower()
    search_dir = '/kaggle/input' if os.path.exists('/kaggle') else '.'
    for root, dirs, files in os.walk(search_dir):
        # Mutate dirs in-place to skip scanning these directories
        dirs[:] = [d for d in dirs if d not in ['.venv', '.venv311', '.git', '.github', '__pycache__', 'node_modules']]
        for f in files:
            if f.lower() == name_lower:
                return os.path.join(root, f)
                
    # Search one level up if local and not found (e.g. if running inside notebooks_standalone/)
    for root, dirs, files in os.walk('..'):
        dirs[:] = [d for d in dirs if d not in ['.venv', '.venv311', '.git', '.github', '__pycache__', 'node_modules']]
        for f in files:
            if f.lower() == name_lower:
                return os.path.join(root, f)
                
    raise FileNotFoundError(f"'{name}' not found.")

print("[OK] Environment setup complete.")
print(f"Kaggle: {KAGGLE} | Inputs: {BASE_IN} | Reports: {REPORT}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Analyze BGL Dataset (Line-level)
# ─────────────────────────────────────────────────────────────────────────────
t0 = time.time()
bgl_path = find_file('BGL_Drain.csv')
print(f"\n[STATS] Analyzing BGL Dataset: {bgl_path}")

bgl_total_rows = 0
bgl_anomalies  = 0
bgl_labels_dict = {}
bgl_templates   = {}
bgl_template_lengths = []

chunk_size = 500_000
for chunk in pd.read_csv(bgl_path, usecols=['template', 'label'], chunksize=chunk_size, on_bad_lines='skip', low_memory=False):
    chunk['template'] = chunk['template'].fillna('').astype(str)
    chunk['label']    = chunk['label'].fillna('-').astype(str).str.strip()
    
    bgl_total_rows += len(chunk)
    anom_mask = (chunk['label'] != '-')
    bgl_anomalies += anom_mask.sum()
    
    # Track label types (vectorized count)
    for lbl, cnt in chunk['label'].value_counts().items():
        bgl_labels_dict[lbl] = bgl_labels_dict.get(lbl, 0) + cnt
        
    # Track template frequency (vectorized count)
    for tmpl, cnt in chunk['template'].value_counts().items():
        bgl_templates[tmpl] = bgl_templates.get(tmpl, 0) + cnt
        
    # Store length of first 100k templates from each chunk
    lengths = chunk['template'].str.len().tolist()
    bgl_template_lengths.extend(lengths[:100_000])
    
    del chunk; gc.collect()

# Load first 10k rows of BGL to extract raw examples (efficiently)
bgl_sample = pd.read_csv(bgl_path, usecols=['log', 'template', 'label'], nrows=10_000, on_bad_lines='skip', low_memory=False)
bgl_sample['template'] = bgl_sample['template'].fillna('').astype(str)
bgl_sample['label']    = bgl_sample['label'].fillna('-').astype(str).str.strip()

bgl_normal_examples = []
bgl_anom_examples   = []

norms = bgl_sample[bgl_sample['label'] == '-'].head(3)
for _, r in norms.iterrows():
    bgl_normal_examples.append({'log': r['log'], 'template': r['template'], 'label': r['label']})
    
anoms = bgl_sample[bgl_sample['label'] != '-'].head(3)
for _, r in anoms.iterrows():
    bgl_anom_examples.append({'log': r['log'], 'template': r['template'], 'label': r['label']})

del bgl_sample; gc.collect()

bgl_time = time.time() - t0
bgl_size_gb = os.path.getsize(bgl_path) / (1024**3)

print(f"   Processed BGL in {bgl_time:.2f}s")
print(f"   Total lines : {bgl_total_rows:,}")
print(f"   Anomalies   : {bgl_anomalies:,} ({100*bgl_anomalies/bgl_total_rows:.2f}%)")
print(f"   Templates   : {len(bgl_templates):,}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Analyze HDFS Dataset (Session-level)
# ─────────────────────────────────────────────────────────────────────────────
t0 = time.time()
hdfs_path = find_file('HDFS_Drain.csv')
print(f"\n[STATS] Analyzing HDFS Dataset: {hdfs_path}")

# Dynamically check columns to optimize loading
sample_hdfs = pd.read_csv(hdfs_path, nrows=5)
hdfs_cols = sample_hdfs.columns.tolist()
print(f"   Detected HDFS columns: {hdfs_cols}")

use_cols = ['template']
lbl_col = 'Label' if 'Label' in hdfs_cols else ('label' if 'label' in hdfs_cols else None)
if lbl_col:
    use_cols.append(lbl_col)
    
has_block_id = 'BlockId' in hdfs_cols
if has_block_id:
    use_cols.append('BlockId')
else:
    # Need log to extract BlockId via regex
    use_cols.append('log')

hdfs_total_rows = 0
hdfs_row_anomalies = 0
hdfs_templates = {}
hdfs_template_lengths = []

# Session tracking
block_events = {}
block_labels = {}
block_order  = []
block_logs   = {} # Store raw log previews

chunk_size = 500_000
for chunk in pd.read_csv(hdfs_path, usecols=use_cols, chunksize=chunk_size, on_bad_lines='skip', low_memory=False):
    chunk['template'] = chunk['template'].fillna('').astype(str)
    
    # Normalize label
    if lbl_col:
        chunk['_anom'] = (chunk[lbl_col].astype(str).str.strip() != 'Normal').astype(np.int8)
    else:
        chunk['_anom'] = np.zeros(len(chunk), dtype=np.int8)
        
    # Extract Block ID
    if has_block_id:
        chunk['_bid'] = chunk['BlockId'].astype(str).str.strip()
    else:
        chunk['_bid'] = chunk['log'].str.extract(r'(blk_-?\d+)', expand=False)
        chunk = chunk.drop(columns=['log']) # Drop raw log immediately to save memory
        
    chunk = chunk.dropna(subset=['_bid'])
    
    hdfs_total_rows += len(chunk)
    hdfs_row_anomalies += chunk['_anom'].sum()
    
    # Track template frequency
    for tmpl, cnt in chunk['template'].value_counts().items():
        hdfs_templates[tmpl] = hdfs_templates.get(tmpl, 0) + cnt
        
    # Store length of templates
    lengths = chunk['template'].str.len().tolist()
    hdfs_template_lengths.extend(lengths[:100_000])
    
    # Vectorized chunk aggregation using GroupBy (MUCH faster than iterrows)
    grouped = chunk.groupby('_bid')
    chunk_events = grouped['template'].apply(list)
    chunk_anom   = grouped['_anom'].max()
    
    # Update global dictionaries in bulk
    for bid, tmpls in chunk_events.items():
        if bid not in block_events:
            block_events[bid] = []
            block_labels[bid] = 0
            block_order.append(bid)
        block_events[bid].extend(tmpls)
        block_labels[bid] = max(block_labels[bid], int(chunk_anom[bid]))
            
    del chunk, grouped, chunk_events, chunk_anom; gc.collect()

# Decoupled example extractor for HDFS sessions (Fast & Memory safe)
# Loads a small slice of raw logs and maps them to their block sessions
print("   Extracting HDFS raw log examples ...")
hdfs_sample = pd.read_csv(hdfs_path, usecols=['log', lbl_col] + (['BlockId'] if has_block_id else []), nrows=25_000, on_bad_lines='skip', low_memory=False)
if has_block_id:
    hdfs_sample['_bid'] = hdfs_sample['BlockId'].astype(str).str.strip()
else:
    hdfs_sample['_bid'] = hdfs_sample['log'].str.extract(r'(blk_-?\d+)', expand=False)

hdfs_sample = hdfs_sample.dropna(subset=['_bid'])
collected_normal_bids = set()
collected_anom_bids = set()

for bid, grp in hdfs_sample.groupby('_bid'):
    is_anom = (grp[lbl_col].astype(str).str.strip() != 'Normal').any()
    if is_anom:
        if len(collected_anom_bids) < 5 and bid not in collected_anom_bids:
            collected_anom_bids.add(bid)
            block_logs[bid] = grp['log'].head(5).tolist()
    else:
        if len(collected_normal_bids) < 5 and bid not in collected_normal_bids:
            collected_normal_bids.add(bid)
            block_logs[bid] = grp['log'].head(5).tolist()

del hdfs_sample; gc.collect()

hdfs_time = time.time() - t0
hdfs_size_gb = os.path.getsize(hdfs_path) / (1024**3)

n_hdfs_blocks = len(block_order)
y_hdfs = np.array([block_labels[bid] for bid in block_order])
n_hdfs_anom_blocks = y_hdfs.sum()
hdfs_block_lengths = np.array([len(block_events[bid]) for bid in block_order])

print(f"   Processed HDFS in {hdfs_time:.2f}s")
print(f"   Total rows       : {hdfs_total_rows:,}")
print(f"   Row anomalies    : {hdfs_row_anomalies:,} ({100*hdfs_row_anomalies/hdfs_total_rows:.2f}%)")
print(f"   Total sessions   : {n_hdfs_blocks:,}")
print(f"   Anom sessions    : {n_hdfs_anom_blocks:,} ({100*n_hdfs_anom_blocks/n_hdfs_blocks:.2f}%)")
print(f"   Session lengths  : min={hdfs_block_lengths.min()}, max={hdfs_block_lengths.max()}, mean={hdfs_block_lengths.mean():.1f}, median={np.median(hdfs_block_lengths):.0f}")
print(f"   Templates        : {len(hdfs_templates):,}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Analyze Spirit Dataset (Sliding Window)
# ─────────────────────────────────────────────────────────────────────────────
t0 = time.time()
spirit_path = find_file('Spirit_Drain.csv')
print(f"\n[STATS] Analyzing Spirit Dataset: {spirit_path}")

spirit_total_rows = 0
spirit_anomalies  = 0
spirit_templates  = {}
spirit_template_lengths = []

spirit_all_labels = []
spirit_all_templates = []

chunk_size = 500_000
for chunk in pd.read_csv(spirit_path, usecols=['template', 'label'], chunksize=chunk_size, on_bad_lines='skip', low_memory=False):
    chunk['template'] = chunk['template'].fillna('').astype(str)
    chunk['label']    = chunk['label'].fillna('-').astype(str).str.strip()
    
    chunk_anoms = (chunk['label'] != '-')
    spirit_total_rows += len(chunk)
    spirit_anomalies += chunk_anoms.sum()
    
    # Track template frequency
    for tmpl, cnt in chunk['template'].value_counts().items():
        spirit_templates[tmpl] = spirit_templates.get(tmpl, 0) + cnt
        
    lengths = chunk['template'].str.len().tolist()
    spirit_template_lengths.extend(lengths[:100_000])
    
    # Store labels and templates for sliding window simulation
    spirit_all_labels.extend((chunk['label'] != '-').astype(np.int8).tolist())
    spirit_all_templates.extend(chunk['template'].tolist())
    
    del chunk; gc.collect()

# Sliding window statistics in NumPy (vectorized, extremely fast)
WINDOW_SIZE = 20
STEP_SIZE   = 10

n_spirit_total   = len(spirit_all_labels)
n_spirit_windows = (n_spirit_total - WINDOW_SIZE) // STEP_SIZE + 1
spirit_labels_arr = np.array(spirit_all_labels, dtype=np.int8)

# Reshape into sliding window rows in NumPy
shape = (n_spirit_windows, WINDOW_SIZE)
strides = (spirit_labels_arr.strides[0] * STEP_SIZE, spirit_labels_arr.strides[0])
windows = np.lib.stride_tricks.as_strided(spirit_labels_arr, shape=shape, strides=strides)
spirit_win_anom = int((windows.max(axis=1) == 1).sum())

# Extract Spirit examples (first 5k lines is enough to get raw examples)
spirit_sample = pd.read_csv(spirit_path, usecols=['log', 'template', 'label'], nrows=5000, on_bad_lines='skip', low_memory=False)
spirit_sample['template'] = spirit_sample['template'].fillna('').astype(str)
spirit_sample['label']    = spirit_sample['label'].fillna('-').astype(str).str.strip()

spirit_sample_labels = (spirit_sample['label'] != '-').astype(np.int8).tolist()
spirit_sample_templates = spirit_sample['template'].tolist()
spirit_sample_logs = spirit_sample['log'].tolist()

del spirit_sample; gc.collect()

spirit_time = time.time() - t0
spirit_size_gb = os.path.getsize(spirit_path) / (1024**3)

print(f"   Processed Spirit in {spirit_time:.2f}s")
print(f"   Total rows       : {spirit_total_rows:,}")
print(f"   Row anomalies    : {spirit_anomalies:,} ({100*spirit_anomalies/spirit_total_rows:.2f}%)")
print(f"   Sliding windows  : {n_spirit_windows:,} (W={WINDOW_SIZE}, S={STEP_SIZE})")
print(f"   Anom windows     : {spirit_win_anom:,} ({100*spirit_win_anom/n_spirit_windows:.2f}%)")
print(f"   Templates        : {len(spirit_templates):,}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Visualizations
# ─────────────────────────────────────────────────────────────────────────────
print("\n[PLOT] Generating thesis-ready figures …")

# Figure 1: Class Distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Class Distribution per Dataset", fontsize=18, fontweight="bold", y=1.02)

# BGL (Line-level)
bgl_counts = [bgl_total_rows - bgl_anomalies, bgl_anomalies]
axes[0].pie(bgl_counts, labels=["Normal", "Anomaly"], colors=palette_binary, autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2.5, antialiased=True))
axes[0].set_title(f"BGL (Line-level)\n({bgl_total_rows:,} logs)", fontsize=14, fontweight="bold")

# HDFS (Session-level)
hdfs_counts = [n_hdfs_blocks - n_hdfs_anom_blocks, n_hdfs_anom_blocks]
axes[1].pie(hdfs_counts, labels=["Normal", "Anomaly"], colors=palette_binary, autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2.5, antialiased=True))
axes[1].set_title(f"HDFS (Session-level)\n({n_hdfs_blocks:,} sessions)", fontsize=14, fontweight="bold")

# Spirit (Sliding Window)
spirit_counts = [n_spirit_windows - spirit_win_anom, spirit_win_anom]
axes[2].pie(spirit_counts, labels=["Normal", "Anomaly"], colors=palette_binary, autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2.5, antialiased=True))
axes[2].set_title(f"Spirit (Sliding Window)\n({n_spirit_windows:,} windows)", fontsize=14, fontweight="bold")

plt.tight_layout()
fig_dist_path = f"{REPORT}/eda_class_distribution_standalone.png"
plt.savefig(fig_dist_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"   Saved: {fig_dist_path}")


# Figure 2: BGL Anomaly Label Types
# Exclude '-' normal logs, sort remaining
bgl_anom_counts = {k: v for k, v in bgl_labels_dict.items() if k != '-'}
bgl_anom_sorted = sorted(bgl_anom_counts.items(), key=lambda x: x[1], reverse=True)[:10]

if bgl_anom_sorted:
    labels, counts = zip(*bgl_anom_sorted)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels, counts, color=COLOR_ANOMALY, edgecolor="white", height=0.6)
    ax.set_title("BGL — Top 10 Anomaly Label Types", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Occurrences", fontsize=12)
    ax.invert_yaxis()
    sns.despine(left=True, bottom=True)
    
    # Add counts to bar ends
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (max(counts) * 0.01), bar.get_y() + bar.get_height()/2, f'{width:,}',
                va='center', ha='left', fontsize=11, fontweight='semibold')
                
    plt.tight_layout()
    fig_bgl_path = f"{REPORT}/eda_bgl_anomaly_types_standalone.png"
    plt.savefig(fig_bgl_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {fig_bgl_path}")


# Figure 3: Template Length Distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
fig.suptitle("Log Template Character Length Distribution", fontsize=16, fontweight="bold", y=1.02)

datasets_lengths = [
    ("BGL", bgl_template_lengths),
    ("HDFS", hdfs_template_lengths),
    ("Spirit", spirit_template_lengths)
]

for idx, (name, lengths) in enumerate(datasets_lengths):
    ax = axes[idx]
    sns.histplot(lengths, bins=40, color='#9B59B6', edgecolor="white", alpha=0.8, ax=ax, kde=True)
    ax.set_title(f"{name} Dataset", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Characters", fontsize=11)
    ax.set_ylabel("Count" if idx == 0 else "", fontsize=11)
    med = np.median(lengths)
    ax.axvline(med, color='orange', linestyle='--', linewidth=2, label=f"Median={med:.0f}")
    ax.legend(fontsize=11)

plt.tight_layout()
fig_len_path = f"{REPORT}/eda_template_lengths_standalone.png"
plt.savefig(fig_len_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"   Saved: {fig_len_path}")


# Figure 4: HDFS Session Length Distribution
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(hdfs_block_lengths, bins=50, color='#1ABC9C', edgecolor="white", alpha=0.85, kde=True, ax=ax)
ax.set_title("HDFS Session Sequence Length Distribution", fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Events per Session (Sequence Length)", fontsize=12)
ax.set_ylabel("Number of Sessions", fontsize=12)
med_len = np.median(hdfs_block_lengths)
mean_len = np.mean(hdfs_block_lengths)
ax.axvline(med_len, color='orange', linestyle='--', linewidth=2, label=f"Median={med_len:.0f}")
ax.axvline(mean_len, color='red', linestyle=':', linewidth=2, label=f"Mean={mean_len:.1f}")
ax.set_xlim(0, 150) # clamp X axis to show most density (max is 300+)
ax.legend(fontsize=12)
sns.despine()

plt.tight_layout()
fig_hdfs_len_path = f"{REPORT}/eda_session_lengths_standalone.png"
plt.savefig(fig_hdfs_len_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"   Saved: {fig_hdfs_len_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Thesis LaTeX Table Exporter
# ─────────────────────────────────────────────────────────────────────────────
print("\n[LATEX] Generating Thesis LaTeX Table …")

latex_content = f"""% Auto-generated by 01_eda_standalone.py
\\begin{{table}}[htbp]
\\centering
\\caption{{Statistical Summary of the Three Log Datasets Used in the Evaluation}}
\\label{{tab:dataset_statistics}}
\\begin{{tabular}}{{lcccccr}}
\\hline
\\textbf{{Dataset}} & \\textbf{{Granularity}} & \\textbf{{Total Lines}} & \\textbf{{Sessions/Windows}} & \\textbf{{Normals}} & \\textbf{{Anomalies}} & \\textbf{{Anomaly Rate (%)}} \\\\ \\hline
BGL & Line-level & {bgl_total_rows:,} & --- & {bgl_total_rows - bgl_anomalies:,} & {bgl_anomalies:,} & {100*bgl_anomalies/bgl_total_rows:.2f}\\% \\\\
HDFS & Session-level & {hdfs_total_rows:,} & {n_hdfs_blocks:,} & {n_hdfs_blocks - n_hdfs_anom_blocks:,} & {n_hdfs_anom_blocks:,} & {100*n_hdfs_anom_blocks/n_hdfs_blocks:.2f}\\% \\\\
Spirit & Sliding Window & {spirit_total_rows:,} & {n_spirit_windows:,} & {n_spirit_windows - spirit_win_anom:,} & {spirit_win_anom:,} & {100*spirit_win_anom/n_spirit_windows:.2f}\\% \\\\ \\hline
\\end{{tabular}}
\\end{{table}}
"""

print("\n--- Generated LaTeX Table Code ---")
print(latex_content)
print("---------------------------------\n")

# Attempt to write to memoire/tables directory
memoire_table_dir = pathlib.Path("memoire/tables")
if memoire_table_dir.exists():
    table_path = memoire_table_dir / "dataset_stats.tex"
    table_path.write_text(latex_content, encoding='utf-8')
    print(f"[OK] Successfully wrote LaTeX table to: {table_path}")
else:
    # Try parent directory relative search
    memoire_table_dir_parent = pathlib.Path("../memoire/tables")
    if memoire_table_dir_parent.exists():
        table_path = memoire_table_dir_parent / "dataset_stats.tex"
        table_path.write_text(latex_content, encoding='utf-8')
        print(f"[OK] Successfully wrote LaTeX table to: {table_path}")
    else:
        # Save locally in report folder as fallback
        table_path = pathlib.Path(REPORT) / "dataset_stats.tex"
        table_path.write_text(latex_content, encoding='utf-8')
        print(f"[WARN] memoire/tables not found. Wrote table locally to: {table_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Real Value Examples (Preview)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Log Examples Preview (Real Values for Thesis)\n")

print("="*80)
print("BGL EXAMPLES (Line-Level)")
print("="*80)
print("\n--- BGL Normal Log Examples ---")
for idx, ex in enumerate(bgl_normal_examples):
    print(f"Example {idx+1}:")
    print(f"  Label   : {ex['label']}")
    print(f"  Template: {ex['template']}")
    print(f"  Log     : {ex['log']}\n")

print("--- BGL Anomaly Log Examples ---")
for idx, ex in enumerate(bgl_anom_examples):
    print(f"Example {idx+1}:")
    print(f"  Label   : {ex['label']}")
    print(f"  Template: {ex['template']}")
    print(f"  Log     : {ex['log']}\n")

print("="*80)
print("HDFS SESSIONS (Session-Level)")
print("="*80)
# Find a normal and anomaly block
normal_bids = [bid for bid in block_order if block_labels[bid] == 0]
anom_bids   = [bid for bid in block_order if block_labels[bid] == 1]

print("\n--- HDFS Normal Session Examples ---")
for idx, bid in enumerate(normal_bids[:2]):
    print(f"Session {idx+1} (Block ID: {bid}, Length: {len(block_events[bid])} logs):")
    print(f"  First 5 logs:")
    logs_to_print = block_logs.get(bid, [])
    for l in logs_to_print:
        print(f"    - {l}")
    print(f"  First 5 templates:")
    for t in block_events[bid][:5]:
        print(f"    - {t}")
    print()

print("--- HDFS Anomaly Session Examples ---")
for idx, bid in enumerate(anom_bids[:2]):
    print(f"Session {idx+1} (Block ID: {bid}, Length: {len(block_events[bid])} logs):")
    print(f"  First 5 logs:")
    logs_to_print = block_logs.get(bid, [])
    for l in logs_to_print:
        print(f"    - {l}")
    print(f"  First 5 templates:")
    for t in block_events[bid][:5]:
        print(f"    - {t}")
    print()

print("="*80)
print("SPIRIT WINDOWS (Sliding Window Level)")
print("="*80)
# Extract a normal and anomaly window sample from spirit_sample
n_sample_spirit_total   = len(spirit_sample_labels)
n_sample_spirit_windows = (n_sample_spirit_total - WINDOW_SIZE) // STEP_SIZE + 1
spirit_sample_labels_arr = np.array(spirit_sample_labels, dtype=np.int8)

sample_normal_win_idx = -1
sample_anom_win_idx   = -1
for i in range(n_sample_spirit_windows):
    start = i * STEP_SIZE
    end   = start + WINDOW_SIZE
    is_anom = spirit_sample_labels_arr[start:end].max() == 1
    if is_anom and sample_anom_win_idx == -1:
        sample_anom_win_idx = i
    if not is_anom and sample_normal_win_idx == -1:
        sample_normal_win_idx = i
    if sample_normal_win_idx != -1 and sample_anom_win_idx != -1:
        break

print("\n--- Spirit Normal Window Example ---")
if sample_normal_win_idx != -1:
    start = sample_normal_win_idx * STEP_SIZE
    end   = start + WINDOW_SIZE
    print(f"Window {sample_normal_win_idx} (Indices {start} to {end}):")
    for offset, (t, l, log_msg) in enumerate(zip(spirit_sample_templates[start:end], spirit_sample_labels_arr[start:end], spirit_sample_logs[start:end])):
        print(f"  [{offset:2d}] Label: {'Normal' if l==0 else 'Anomaly'} | Template: {t} | Log: {log_msg}")
else:
    print("No normal window found in sample.")

print("\n--- Spirit Anomaly Window Example ---")
if sample_anom_win_idx != -1:
    start = sample_anom_win_idx * STEP_SIZE
    end   = start + WINDOW_SIZE
    print(f"Window {sample_anom_win_idx} (Indices {start} to {end}):")
    for offset, (t, l, log_msg) in enumerate(zip(spirit_sample_templates[start:end], spirit_sample_labels_arr[start:end], spirit_sample_logs[start:end])):
        print(f"  [{offset:2d}] Label: {'Normal' if l==0 else 'Anomaly'} | Template: {t} | Log: {log_msg}")
else:
    print("No anomaly window found in sample.")

print("\n[INFO] Exploratory Data Analysis & Statistics Extraction Completed successfully!")

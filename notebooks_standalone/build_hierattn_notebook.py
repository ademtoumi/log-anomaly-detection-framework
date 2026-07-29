#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
"""
Convert HierAttnBlock_HDFS.py → HierAttnBlock_HDFS.ipynb

Splits the script into notebook cells at the ╔═══ STEP N ═══╗ banners,
adds markdown headers, and writes a valid Kaggle-compatible .ipynb.
"""

import json, re, os

SRC  = "HierAttnBlock_HDFS.py"
DEST = "HierAttnBlock_HDFS.ipynb"

# Read source
with open(SRC, encoding="utf-8") as f:
    src = f.read()

# --- Split into named sections -----------------------------------------------
# Each section starts with a line containing ╔═══ ... ═══╗
SECTION_RE = re.compile(
    r'^# [╔╚]═+╗?\n?# .*\n# [╔╚]═+╝?\n',
    re.MULTILINE
)

# Find all banner positions
banner_spans = [(m.start(), m.end(), m.group()) for m in SECTION_RE.finditer(src)]

# Helper to extract readable title from banner text
def banner_title(text):
    for line in text.split("\n"):
        inner = line.strip("# ╔╗╚╝═").strip()
        if inner:
            return inner
    return "Section"

sections = []
for i, (start, end, btext) in enumerate(banner_spans):
    title = banner_title(btext)
    code_start = end
    code_end   = banner_spans[i+1][0] if i+1 < len(banner_spans) else len(src)
    code       = src[code_start:code_end].strip("\n")
    sections.append({"title": title, "code": code, "banner": btext})

# If no banners found, put everything in one cell
if not sections:
    sections = [{"title": "Full Pipeline", "code": src, "banner": ""}]

# --- Build notebook cells ----------------------------------------------------
cells = []

# Title markdown cell
cells.append({
    "cell_type": "markdown",
    "id": "cell_title",
    "metadata": {},
    "source": [
        "# HierAttn-Block — HDFS Log Anomaly Detection\n",
        "\n",
        "**Full pipeline**: Steps 1–11  \n",
        "**Dataset**: HDFS_Drain.csv (Drain-parsed, columns: log, label, template)  \n",
        "**Goal**: Beat DeepLog and LogBERT baselines; produce thesis-ready results.\n",
        "\n",
        "---",
    ]
})

for i, sec in enumerate(sections):
    # Markdown header for each step
    cells.append({
        "cell_type": "markdown",
        "id": f"md_step_{i}",
        "metadata": {},
        "source": [f"## {sec['title']}\n"]
    })

    # Code cell — split into lines with \n endings
    code_lines = [l + "\n" for l in sec["code"].split("\n")]
    # Remove trailing blank lines
    while code_lines and code_lines[-1].strip() == "":
        code_lines.pop()

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": f"code_step_{i}",
        "metadata": {},
        "outputs": [],
        "source": code_lines
    })

# --- Assemble notebook -------------------------------------------------------
notebook = {
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
            "version": "3.10.0"
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [
                {
                    "sourceType": "datasetVersion",
                    "sourceId": "yahiachammemi/logs-drain-datasets-hdfs-bgl-spirit",
                    "datasetId": "logs-drain-datasets-hdfs-bgl-spirit"
                }
            ],
            "dockerImageVersionId": 30823,
            "isInternetEnabled": False,
            "language": "python",
            "isGpuEnabled": True
        }
    },
    "cells": cells
}

dest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEST)
with open(dest_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"[OK] Notebook written: {dest_path}")
print(f"     Cells: {len(cells)}")

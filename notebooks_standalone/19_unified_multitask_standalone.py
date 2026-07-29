# =============================================================================
# STANDALONE NOTEBOOK 19 — Unified Multi-Task Log Anomaly Detection (Kaggle Standalone)
#
# ✅ ZERO external package dependencies — runs fully self-contained on Kaggle.
# ✅ Dynamic find_file helper — locates raw CSV logs in any Kaggle input directory.
# ✅ Memory-Safe Data Loading — pre-tokenizes and shards data to Parquet.
# ✅ Stage 1 Masked Language Modeling (MLM) pretraining of shared DistilBERT encoder.
# ✅ Stage 2 Joint Multi-Task training with WeightedRandomSampler balancing,
#    mixed-precision (fp16), gradient accumulation, and RAM watchdog.
# ✅ F1-sensitive percentile-bounded HDFS threshold grid search.
# ✅ Integrated SHAP GradientExplainer attributions and HDFS MSE heatmaps.
# =============================================================================

import os
import gc
import re
import json
import time
import random
import pathlib
import warnings
import numpy as np
import pandas as pd
import joblib
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score

# Hugging Face imports
from transformers import (
    DistilBertConfig, DistilBertModel, DistilBertForMaskedLM,
    get_cosine_schedule_with_warmup
)
import shap

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Seed for reproducibility ─────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Environment & Path Setup ─────────────────────────────────────────────────
KAGGLE = os.path.exists('/kaggle')
BASE_OUT = '/kaggle/working' if KAGGLE else 'result/unified_standalone'
PROCESSED_DIR = f'{BASE_OUT}/processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Running on: {DEVICE} | Environment: {'Kaggle' if KAGGLE else 'Local'}")

# Global Hyperparameters
CONFIG = {
    "limit_rows": 500000,
    "max_len": 128,
    "mlm_epochs": 3,
    "mlm_lr": 1e-4,
    "mt_epochs": 8,          # FIX: was 10 — 8 epochs keeps total runtime under Kaggle limit
    "mt_batch_size": 8,
    "mt_grad_accum": 4,
    "mt_lr": 2e-5,
    "mt_weight_decay": 0.01,
    "mt_warmup_ratio": 0.1,
    "mt_patience": 4,         # FIX: was 5 — stops faster when no improvement
    "optuna_trials": 1,       # FIX: was 5 — 5 trials = 5x full training = Kaggle timeout
    "optuna_enabled": False   # FIX: was True — disable Optuna, use fixed equal weights
                               # Each trial takes ~80 min/epoch * 8 epochs = 10+ hours total
                               # Use optuna_enabled=True ONLY when running locally without limit
}

# Checkpoint system
CKPT = pathlib.Path(BASE_OUT) / 'ckpt_19_unified_multitask.json'

def save_ckpt(d):
    with open(CKPT, 'w') as f:
        json.dump(d, f)

def load_ckpt():
    if CKPT.exists():
        with open(CKPT) as f:
            return json.load(f)
    return {}

ckpt_state = load_ckpt()

# Special tokens
PAD_IDX = 0
CLS_IDX = 1
UNK_IDX = 2
MASK_IDX = 3

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Helpers & Data Finding
# ─────────────────────────────────────────────────────────────────────────────
def find_file(name):
    name_lower = name.lower()
    search_dir = '/kaggle/input' if KAGGLE else '.'
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.lower() == name_lower:
                return os.path.join(root, f)
    # Fallback search locally
    for root, _, files in os.walk('.'):
        for f in files:
            if f.lower() == name_lower:
                return os.path.join(root, f)
    raise FileNotFoundError(f"Could not find dataset file: '{name}'")

def ram_watchdog(step_name=""):
    used_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
    if used_gb > 22.0:
        print(f"[Watchdog] RAM used: {used_gb:.2f}GB at '{step_name}'. Cleared memory.")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Data Preprocessing, Splitting and Sharding (Task 3.1)
# ─────────────────────────────────────────────────────────────────────────────
if 'preprocessing_done' not in ckpt_state:
    print("\n" + "="*80)
    print("  [CELL 2] Running Data Preprocessing and Parquet Sharding...")
    print("="*80)
    
    # 2.1 HDFS block session parsing
    hdfs_file = find_file('HDFS_Drain.csv')
    block_events = {}
    block_labels = {}
    block_order = []
    
    print("Loading and sessionizing HDFS logs...")
    for chunk in pd.read_csv(hdfs_file, chunksize=100000, on_bad_lines='skip', low_memory=False):
        if len(block_events) >= 100000:  # limit to prevent notebook timeout
            break
        if 'BlockId' in chunk.columns:
            chunk['_bid'] = chunk['BlockId'].astype(str).str.strip()
        else:
            chunk['_bid'] = chunk['log'].astype(str).str.extract(r'(blk_-?\d+)')
            
        chunk = chunk.dropna(subset=['_bid'])
        lbl_col = 'Label' if 'Label' in chunk.columns else 'label'
        chunk['_anom'] = (chunk[lbl_col].astype(str).str.strip() != 'Normal').astype(int)
        
        for _, row in chunk[['_bid', 'template', '_anom']].iterrows():
            bid = row['_bid']
            if bid not in block_events:
                block_events[bid] = []
                block_labels[bid] = 0
                block_order.append(bid)
            tmpl = str(row['template']) if pd.notna(row['template']) else 'unknown'
            block_events[bid].append(tmpl)
            block_labels[bid] = max(block_labels[bid], int(row['_anom']))
            
        del chunk
        gc.collect()
        
    hdfs_samples = [{"id": bid, "templates": block_events[bid], "label": block_labels[bid]} for bid in block_order]
    
    # 2.2 BGL line-level parsing
    bgl_file = find_file('BGL_Drain.csv')
    print("Loading BGL logs...")
    bgl_df = pd.read_csv(bgl_file, nrows=CONFIG["limit_rows"], usecols=['label', 'template'], on_bad_lines='skip', low_memory=False)
    bgl_df['template'] = bgl_df['template'].fillna('unknown').astype(str)
    bgl_df['label'] = bgl_df['label'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
    bgl_samples = [{"id": f"bgl_{idx}", "templates": [row["template"]], "label": int(row["label"])} for idx, row in bgl_df.iterrows()]
    del bgl_df
    gc.collect()
    
    # 2.3 Spirit sliding window parsing
    spirit_file = find_file('Spirit_Drain.csv')
    print("Loading Spirit logs...")
    spirit_df = pd.read_csv(spirit_file, nrows=CONFIG["limit_rows"], usecols=['label', 'template'], on_bad_lines='skip', low_memory=False)
    spirit_df['template'] = spirit_df['template'].fillna('unknown').astype(str)
    spirit_df['label'] = spirit_df['label'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
    
    templates = spirit_df['template'].values
    labels = spirit_df['label'].values
    spirit_samples = []
    for start_idx in range(0, len(templates) - 20 + 1, 10):
        win_templates = templates[start_idx : start_idx + 20].tolist()
        win_labels = labels[start_idx : start_idx + 20]
        win_label = int(np.any(win_labels == 1))
        spirit_samples.append({"id": f"spirit_{start_idx}", "templates": win_templates, "label": win_label})
        
    del spirit_df
    gc.collect()
    
    # 2.4 Splitting datasets (Zero-Leakage Splitting)
    # HDFS - Temporal
    n_hdfs = len(hdfs_samples)
    i1_hdfs, i2_hdfs = int(n_hdfs * 0.80), int(n_hdfs * 0.90)
    hdfs_train, hdfs_val, hdfs_test = hdfs_samples[:i1_hdfs], hdfs_samples[i1_hdfs:i2_hdfs], hdfs_samples[i2_hdfs:]
    
    # BGL - Stratified
    bgl_labels = [s["label"] for s in bgl_samples]
    bgl_train_idx, bgl_temp_idx = train_test_split(np.arange(len(bgl_samples)), test_size=0.20, stratify=bgl_labels, random_state=SEED)
    bgl_temp_lbls = [bgl_samples[i]["label"] for i in bgl_temp_idx]
    bgl_val_idx, bgl_test_idx = train_test_split(bgl_temp_idx, test_size=0.50, stratify=bgl_temp_lbls, random_state=SEED)
    
    bgl_train = [bgl_samples[i] for i in bgl_train_idx]
    bgl_val = [bgl_samples[i] for i in bgl_val_idx]
    bgl_test = [bgl_samples[i] for i in bgl_test_idx]
    
    # Spirit - Temporal
    n_spirit = len(spirit_samples)
    i1_spirit, i2_spirit = int(n_spirit * 0.80), int(n_spirit * 0.90)
    spirit_train, spirit_val, spirit_test = spirit_samples[:i1_spirit], spirit_samples[i1_spirit:i2_spirit], spirit_samples[i2_spirit:]
    
    # 2.5 Fit Vocabulary on TRAINING split only (no-leakage)
    print("Fitting unified vocabulary...")
    train_templates = set()
    for s in hdfs_train: train_templates.update(s["templates"])
    for s in bgl_train: train_templates.update(s["templates"])
    for s in spirit_train: train_templates.update(s["templates"])
    
    vocab = {"[PAD]": PAD_IDX, "[CLS]": CLS_IDX, "[UNK]": UNK_IDX, "[MASK]": MASK_IDX}
    for idx, temp in enumerate(sorted(train_templates)):
        vocab[temp] = idx + 4
        
    joblib.dump(vocab, f'{BASE_OUT}/vocab.pkl')
    print(f"  Fitted vocabulary size: {len(vocab):,}")
    
    # 2.6 Helper to tokenize and shard Parquet
    def save_split_parquet(samples, vocab, dataset_id, name):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        for sample in samples:
            ids = [CLS_IDX] + [vocab.get(t, UNK_IDX) for t in sample["templates"]]
            if len(ids) > CONFIG["max_len"]:
                ids = ids[:CONFIG["max_len"]]
            padding = CONFIG["max_len"] - len(ids)
            attention_mask = [False]*len(ids) + [True]*padding
            ids = ids + [PAD_IDX]*padding
            
            input_ids_list.append(ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(float(sample["label"]))
            
        df = pd.DataFrame({
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "label": labels_list,
            "dataset_id": [dataset_id] * len(samples)
        })
        df.to_parquet(f'{PROCESSED_DIR}/{name}.parquet', engine='pyarrow')
        
    print("Saving processed Parquet splits to disk...")
    save_split_parquet(hdfs_train, vocab, 0, 'hdfs_train')
    save_split_parquet(hdfs_val, vocab, 0, 'hdfs_val')
    save_split_parquet(hdfs_test, vocab, 0, 'hdfs_test')
    
    save_split_parquet(bgl_train, vocab, 1, 'bgl_train')
    save_split_parquet(bgl_val, vocab, 1, 'bgl_val')
    save_split_parquet(bgl_test, vocab, 1, 'bgl_test')
    
    save_split_parquet(spirit_train, vocab, 2, 'spirit_train')
    save_split_parquet(spirit_val, vocab, 2, 'spirit_val')
    save_split_parquet(spirit_test, vocab, 2, 'spirit_test')
    
    ckpt_state['preprocessing_done'] = True
    save_ckpt(ckpt_state)
    print("  Preprocessing complete!")
else:
    print("  [Step 2] Loaded from checkpoint (shards exist).")
    vocab = joblib.load(f'{BASE_OUT}/vocab.pkl')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — PyTorch Dataset Loading (Task 3.2 dataset wrapper)
# ─────────────────────────────────────────────────────────────────────────────
class TokenizedDataset(Dataset):
    def __init__(self, filepath: str):
        super().__init__()
        df = pd.read_parquet(filepath)
        self.input_ids = torch.tensor(np.stack(df['input_ids'].values), dtype=torch.long)
        self.attention_mask = torch.tensor(np.stack(df['attention_mask'].values), dtype=torch.bool)
        self.labels = torch.tensor(df['label'].values, dtype=torch.float)
        self.dataset_ids = torch.tensor(df['dataset_id'].values, dtype=torch.long)
        del df
        gc.collect()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
            "dataset_id": self.dataset_ids[idx]
        }

def get_balanced_sampler(concat_dataset: ConcatDataset) -> WeightedRandomSampler:
    dataset_sizes = [len(d) for d in concat_dataset.datasets]
    class_weights = [1.0 / (3.0 * size) if size > 0 else 0.0 for size in dataset_sizes]
    sample_weights = []
    for d_idx, dataset in enumerate(concat_dataset.datasets):
        sample_weights.extend([class_weights[d_idx]] * len(dataset))
    return WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Model Architecture Components (Task 3.3 models)
# ─────────────────────────────────────────────────────────────────────────────
class SharedLogEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 768, max_len: int = 128):
        super().__init__()
        self.config = DistilBertConfig(
            vocab_size=vocab_size, dim=embed_dim, n_layers=6, n_heads=12,
            hidden_dim=embed_dim*4, pad_token_id=0, max_position_embeddings=max_len
        )
        self.distilbert = DistilBertModel(self.config)
        self.distilbert.gradient_checkpointing_enable()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        hf_mask = (~attention_mask).long() if attention_mask is not None else None
        outputs = self.distilbert(input_ids=input_ids, attention_mask=hf_mask)
        token_embeddings = outputs.last_hidden_state
        return token_embeddings, token_embeddings[:, 0, :]

class HDFSReconstructionHead(nn.Module):
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256, latent_dim: int = 512):
        super().__init__()
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.bottleneck = nn.Sequential(nn.Linear(hidden_dim * 2, latent_dim), nn.ReLU())
        self.decoder = nn.LSTM(latent_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        _, (h_n, _) = self.encoder(x)
        context = torch.cat((h_n[-2], h_n[-1]), dim=-1) # final bidirectional states
        latent = self.bottleneck(context)
        dec_input = latent.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(dec_input)
        return self.proj(decoded)

class BGLClassificationHead(nn.Module):
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(embed_dim, 1))
    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        return self.fc(cls_emb).squeeze(-1)

class SpiritClassificationHead(nn.Module):
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(embed_dim, 1))

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is not None:
            mask = (~attention_mask).float().unsqueeze(-1)
            pooled = torch.sum(x * mask, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1.0)
        else:
            pooled = torch.mean(x, dim=1)
        return self.fc(pooled).squeeze(-1)

class UnifiedLogAnomalyModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 768, max_len: int = 128):
        super().__init__()
        self.encoder = SharedLogEncoder(vocab_size, embed_dim, max_len)
        self.hdfs_head = HDFSReconstructionHead(embed_dim)
        self.bgl_head = BGLClassificationHead(embed_dim)
        self.spirit_head = SpiritClassificationHead(embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, dataset_id: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        token_embs, cls_emb = self.encoder(input_ids, attention_mask)
        outputs = {}
        
        if dataset_id is None:
            outputs["hdfs_embeddings"] = token_embs
            outputs["hdfs_reconstructed"] = self.hdfs_head(token_embs)
            outputs["bgl_logits"] = self.bgl_head(cls_emb)
            outputs["spirit_logits"] = self.spirit_head(token_embs, attention_mask)
            return outputs

        h_mask, b_mask, s_mask = (dataset_id == 0), (dataset_id == 1), (dataset_id == 2)
        if h_mask.any():
            hdfs_toks = token_embs[h_mask]
            outputs["hdfs_embeddings"] = hdfs_toks
            outputs["hdfs_reconstructed"] = self.hdfs_head(hdfs_toks)
        if b_mask.any():
            outputs["bgl_logits"] = self.bgl_head(cls_emb[b_mask])
        if s_mask.any():
            spirit_toks = token_embs[s_mask]
            att_subset = attention_mask[s_mask] if attention_mask is not None else None
            outputs["spirit_logits"] = self.spirit_head(spirit_toks, att_subset)
            
        return outputs

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Stage-1 MLM Pretraining (Task 3.2 MLM)
# ─────────────────────────────────────────────────────────────────────────────
class MLMDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, vocab_size: int, mask_prob: float = 0.15):
        self.input_ids = input_ids
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = self.input_ids[idx].clone()
        labels = input_ids.clone()
        prob_matrix = torch.full(input_ids.shape, self.mask_prob)
        special_mask = (input_ids == PAD_IDX) | (input_ids == CLS_IDX)
        prob_matrix.masked_fill_(special_mask, value=0.0)
        
        masked_indices = torch.bernoulli(prob_matrix).bool()
        labels[~masked_indices] = -100
        
        # 80% [MASK], 10% random, 10% keep
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = MASK_IDX
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(4, self.vocab_size, input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_tokens[indices_random]
        return input_ids, labels

if 'mlm_pretrain_done' not in ckpt_state:
    print("\n" + "="*80)
    print("  [CELL 5] Stage-1 MLM Shared Encoder Pretraining...")
    print("="*80)
    
    # Load all training ids
    train_ids = []
    for d in ["hdfs", "bgl", "spirit"]:
        df = pd.read_parquet(f'{PROCESSED_DIR}/{d}_train.parquet')
        train_ids.extend(df["input_ids"].tolist())
    train_ids_tensor = torch.tensor(train_ids, dtype=torch.long)
    
    mlm_ds = MLMDataset(train_ids_tensor, len(vocab))
    mlm_loader = DataLoader(mlm_ds, batch_size=64, shuffle=True)
    
    mlm_config = DistilBertConfig(
        vocab_size=len(vocab), dim=768, n_layers=6, n_heads=12,
        hidden_dim=3072, pad_token_id=0, max_position_embeddings=CONFIG["max_len"]
    )
    mlm_model = DistilBertForMaskedLM(mlm_config).to(DEVICE)
    mlm_model.distilbert.gradient_checkpointing_enable()
    
    optimizer = optim.AdamW(mlm_model.parameters(), lr=CONFIG["mlm_lr"], weight_decay=0.01)
    scaler = amp.GradScaler()
    
    mlm_model.train()
    for epoch in range(1, CONFIG["mlm_epochs"] + 1):
        total_loss = 0.0
        for b_ids, b_labels in mlm_loader:
            b_ids, b_labels = b_ids.to(DEVICE), b_labels.to(DEVICE)
            optimizer.zero_grad()
            with amp.autocast():
                loss = mlm_model(input_ids=b_ids, labels=b_labels).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"  MLM Epoch {epoch} Loss: {total_loss / len(mlm_loader):.4f}")
        
    torch.save(mlm_model.distilbert.state_dict(), f'{BASE_OUT}/distilbert_mlm.pt')
    ckpt_state['mlm_pretrain_done'] = True
    save_ckpt(ckpt_state)
    print("  Stage-1 MLM Pretraining complete!")
    del mlm_model, optimizer
    gc.collect()
else:
    print("  [Step 5] MLM pretraining weights found on disk.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Stage-2 Loss Formulations (Task 3.3 losses)
# ─────────────────────────────────────────────────────────────────────────────
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
    def forward(self, pred, target, mask):
        active = (~mask).float().unsqueeze(-1)
        squared = self.mse(pred, target) * active
        return squared.sum() / torch.clamp(active.sum() * pred.size(-1), min=1.0)

class MultiTaskLoss(nn.Module):
    def __init__(self, l1, l2, l3, w_bgl, w_sp):
        super().__init__()
        self.l1, self.l2, self.l3 = l1, l2, l3
        self.hdfs_crit = MaskedMSELoss()
        self.bgl_crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w_bgl]))
        self.spirit_crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w_sp]))

    def forward(self, preds, targets, dataset_ids, attention_mask=None):
        device = dataset_ids.device
        loss_total = torch.tensor(0.0, device=device)
        losses_log = {}
        
        h_m, b_m, s_m = (dataset_ids == 0), (dataset_ids == 1), (dataset_ids == 2)
        
        # 1. HDFS (unsupervised, normals only)
        if h_m.any() and "hdfs_reconstructed" in preds:
            norm_h = (targets[h_m] == 0)
            if norm_h.any():
                hdfs_recon = preds["hdfs_reconstructed"][norm_h]
                hdfs_embed = preds["hdfs_embeddings"][norm_h]
                hdfs_att_mask = attention_mask[h_m][norm_h] if attention_mask is not None else torch.zeros_like(hdfs_recon[:,:,0]).bool()
                loss_h = self.hdfs_crit(hdfs_recon, hdfs_embed, hdfs_att_mask)
                loss_total += self.l1 * loss_h
                losses_log["loss_hdfs"] = loss_h.item()
        # 2. BGL
        if b_m.any() and "bgl_logits" in preds:
            self.bgl_crit.pos_weight = self.bgl_crit.pos_weight.to(device)
            loss_b = self.bgl_crit(preds["bgl_logits"], targets[b_m].float())
            loss_total += self.l2 * loss_b
            losses_log["loss_bgl"] = loss_b.item()
        # 3. Spirit
        if s_m.any() and "spirit_logits" in preds:
            self.spirit_crit.pos_weight = self.spirit_crit.pos_weight.to(device)
            loss_s = self.spirit_crit(preds["spirit_logits"], targets[s_m].float())
            loss_total += self.l3 * loss_s
            losses_log["loss_spirit"] = loss_s.item()
            
        losses_log["loss_total"] = loss_total.item()
        return loss_total, losses_log

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Threshold Tuning (Task 3.4)
# ─────────────────────────────────────────────────────────────────────────────
def tune_hdfs_threshold(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    all_errors, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            d_id = torch.zeros(ids.size(0), dtype=torch.long, device=device)
            
            outputs = model(ids, mask, d_id)
            pred = outputs["hdfs_reconstructed"]
            target = outputs["hdfs_embeddings"]
            
            active = (~mask).float().unsqueeze(-1)
            sq_err = ((pred - target) ** 2) * active
            seq_mse = sq_err.sum(dim=(1, 2)) / torch.clamp(active.sum(dim=(1, 2)) * pred.size(-1), min=1.0)
            
            all_errors.extend(seq_mse.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())
            
    errors = np.array(all_errors)
    targets = np.array(all_targets, dtype=int)
    
    min_tr = np.percentile(errors, 1.0)
    max_tr = np.percentile(errors, 99.0)
    
    thresholds = np.linspace(min_tr, max_tr, 200)
    best_f1, best_tr = 0.0, float(min_tr)
    best_p, best_r = 0.0, 0.0
    
    for tr in thresholds:
        preds = (errors > tr).astype(int)
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tr = float(tr)
            best_p = float(precision_score(targets, preds, zero_division=0))
            best_r = float(recall_score(targets, preds, zero_division=0))
            
    return best_tr, best_f1, {"precision": best_p, "recall": best_r}

def evaluate_test_set(model, test_hdfs, test_bgl, test_spirit, hdfs_threshold, device) -> pd.DataFrame:
    """Evaluates the multi-task model on HDFS, BGL, and Spirit test partitions,
    computing Precision, Recall, F1, MCC, and AUC-ROC for each dataset.
    """
    model.eval()
    
    # 1. HDFS Test Set
    hdfs_loader = DataLoader(test_hdfs, batch_size=128, shuffle=False)
    hdfs_errors, hdfs_targets = [], []
    with torch.no_grad():
        for batch in hdfs_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            d_id = torch.zeros(ids.size(0), dtype=torch.long, device=device)
            
            outputs = model(ids, mask, d_id)
            pred = outputs["hdfs_reconstructed"]
            target = outputs["hdfs_embeddings"]
            
            active = (~mask).float().unsqueeze(-1)
            sq_err = ((pred - target) ** 2) * active
            seq_mse = sq_err.sum(dim=(1, 2)) / torch.clamp(active.sum(dim=(1, 2)) * pred.size(-1), min=1.0)
            
            hdfs_errors.extend(seq_mse.cpu().numpy().tolist())
            hdfs_targets.extend(labels.numpy().tolist())
            
    hdfs_errors = np.array(hdfs_errors)
    hdfs_preds = (hdfs_errors > hdfs_threshold).astype(int)
    hdfs_targets = np.array(hdfs_targets, dtype=int)
    
    hdfs_metrics = {
        "Dataset": "HDFS",
        "Precision": precision_score(hdfs_targets, hdfs_preds, zero_division=0),
        "Recall": recall_score(hdfs_targets, hdfs_preds, zero_division=0),
        "F1": f1_score(hdfs_targets, hdfs_preds, zero_division=0),
        "MCC": matthews_corrcoef(hdfs_targets, hdfs_preds),
        "AUC-ROC": roc_auc_score(hdfs_targets, hdfs_errors)
    }
    
    # 2. BGL Test Set
    bgl_loader = DataLoader(test_bgl, batch_size=128, shuffle=False)
    bgl_probs, bgl_targets = [], []
    with torch.no_grad():
        for batch in bgl_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            d_id = torch.ones(ids.size(0), dtype=torch.long, device=device)
            
            outputs = model(ids, mask, d_id)
            probs = torch.sigmoid(outputs["bgl_logits"]).cpu().numpy()
            
            bgl_probs.extend(probs.tolist())
            bgl_targets.extend(labels.numpy().tolist())
            
    bgl_probs = np.array(bgl_probs)
    bgl_preds = (bgl_probs >= 0.5).astype(int)
    bgl_targets = np.array(bgl_targets, dtype=int)
    
    bgl_metrics = {
        "Dataset": "BGL",
        "Precision": precision_score(bgl_targets, bgl_preds, zero_division=0),
        "Recall": recall_score(bgl_targets, bgl_preds, zero_division=0),
        "F1": f1_score(bgl_targets, bgl_preds, zero_division=0),
        "MCC": matthews_corrcoef(bgl_targets, bgl_preds),
        "AUC-ROC": roc_auc_score(bgl_targets, bgl_probs)
    }
    
    # 3. Spirit Test Set
    spirit_loader = DataLoader(test_spirit, batch_size=128, shuffle=False)
    spirit_probs, spirit_targets = [], []
    with torch.no_grad():
        for batch in spirit_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            d_id = torch.ones(ids.size(0), dtype=torch.long, device=device) * 2
            
            outputs = model(ids, mask, d_id)
            probs = torch.sigmoid(outputs["spirit_logits"]).cpu().numpy()
            
            spirit_probs.extend(probs.tolist())
            spirit_targets.extend(labels.numpy().tolist())
            
    spirit_probs = np.array(spirit_probs)
    spirit_preds = (spirit_probs >= 0.5).astype(int)
    spirit_targets = np.array(spirit_targets, dtype=int)
    
    spirit_metrics = {
        "Dataset": "Spirit",
        "Precision": precision_score(spirit_targets, spirit_preds, zero_division=0),
        "Recall": recall_score(spirit_targets, spirit_preds, zero_division=0),
        "F1": f1_score(spirit_targets, spirit_preds, zero_division=0),
        "MCC": matthews_corrcoef(spirit_targets, spirit_preds),
        "AUC-ROC": roc_auc_score(spirit_targets, spirit_probs)
    }
    
    results_df = pd.DataFrame([hdfs_metrics, bgl_metrics, spirit_metrics])
    return results_df

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Joint Multi-Task Trainer Setup (Task 3.3)
# ─────────────────────────────────────────────────────────────────────────────
class MultiTaskTrainer:
    def __init__(self):
        self.device = DEVICE
        
        # Load processed datasets
        self.train_hdfs = TokenizedDataset(f'{PROCESSED_DIR}/hdfs_train.parquet')
        self.train_bgl = TokenizedDataset(f'{PROCESSED_DIR}/bgl_train.parquet')
        self.train_spirit = TokenizedDataset(f'{PROCESSED_DIR}/spirit_train.parquet')
        
        self.val_hdfs = TokenizedDataset(f'{PROCESSED_DIR}/hdfs_val.parquet')
        self.val_bgl = TokenizedDataset(f'{PROCESSED_DIR}/bgl_val.parquet')
        self.val_spirit = TokenizedDataset(f'{PROCESSED_DIR}/spirit_val.parquet')
        
        self.train_concat = ConcatDataset([self.train_hdfs, self.train_bgl, self.train_spirit])

    def evaluate_val(self, model):
        model.eval()
        ram_watchdog("Val Loop Start")
        
        # 1. HDFS Val
        h_tr, h_f1, _ = tune_hdfs_threshold(model, self.val_hdfs, self.device)
        
        # 2. BGL Val
        bgl_loader = DataLoader(self.val_bgl, batch_size=128, shuffle=False)
        bgl_probs, bgl_targets = [], []
        with torch.no_grad():
            for batch in bgl_loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                d_id = torch.ones(ids.size(0), dtype=torch.long, device=self.device)
                probs = torch.sigmoid(model(ids, mask, d_id)["bgl_logits"]).cpu().numpy()
                bgl_probs.extend(probs.tolist())
                bgl_targets.extend(batch["label"].numpy().tolist())
        bgl_f1 = f1_score(bgl_targets, (np.array(bgl_probs) >= 0.5).astype(int), zero_division=0)
        
        # 3. Spirit Val
        spirit_loader = DataLoader(self.val_spirit, batch_size=128, shuffle=False)
        spirit_probs, spirit_targets = [], []
        with torch.no_grad():
            for batch in spirit_loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                d_id = torch.ones(ids.size(0), dtype=torch.long, device=self.device) * 2
                probs = torch.sigmoid(model(ids, mask, d_id)["spirit_logits"]).cpu().numpy()
                spirit_probs.extend(probs.tolist())
                spirit_targets.extend(batch["label"].numpy().tolist())
        spirit_f1 = f1_score(spirit_targets, (np.array(spirit_probs) >= 0.5).astype(int), zero_division=0)
        
        return {"hdfs_f1": h_f1, "bgl_f1": bgl_f1, "spirit_f1": spirit_f1, "mean_f1": (h_f1 + bgl_f1 + spirit_f1)/3.0, "threshold": h_tr}

    def train_model(self, l1, l2, l3) -> float:
        model = UnifiedLogAnomalyModel(vocab_size=len(vocab), max_len=CONFIG["max_len"]).to(self.device)
        model.encoder.distilbert.load_state_dict(torch.load(f'{BASE_OUT}/distilbert_mlm.pt', map_location=self.device))
        
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["mt_lr"], weight_decay=CONFIG["mt_weight_decay"])
        sampler = get_balanced_sampler(self.train_concat)
        train_loader = DataLoader(self.train_concat, batch_size=CONFIG["mt_batch_size"], sampler=sampler, num_workers=2, pin_memory=True)
        
        total_steps = len(train_loader) * CONFIG["mt_epochs"] // CONFIG["mt_grad_accum"]
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps*CONFIG["mt_warmup_ratio"]), total_steps)
        scaler = amp.GradScaler()
        
        criterion = MultiTaskLoss(l1, l2, l3, 1.415, 2.3)
        
        best_mean_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(1, CONFIG["mt_epochs"] + 1):
            model.train()
            optimizer.zero_grad()
            total_loss = 0.0
            
            for idx, batch in enumerate(train_loader):
                ram_watchdog(f"Train Step {idx}")
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                d_ids = batch["dataset_id"].to(self.device)
                
                with amp.autocast():
                    preds = model(ids, mask, d_ids)
                    loss, _ = criterion(preds, labels, d_ids, mask)
                    loss = loss / CONFIG["mt_grad_accum"]
                    
                scaler.scale(loss).backward()
                
                if (idx + 1) % CONFIG["mt_grad_accum"] == 0 or (idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    
                total_loss += loss.item() * CONFIG["mt_grad_accum"]
                
            val_metrics = self.evaluate_val(model)
            print(f"Epoch {epoch:02d} | Train Loss: {total_loss/len(train_loader):.4f} | Mean Val F1: {val_metrics['mean_f1']:.4f} (HDFS: {val_metrics['hdfs_f1']:.4f}, BGL: {val_metrics['bgl_f1']:.4f}, Spirit: {val_metrics['spirit_f1']:.4f})")
            
            if val_metrics["mean_f1"] > best_mean_f1:
                best_mean_f1 = val_metrics["mean_f1"]
                patience_counter = 0
                torch.save(model.state_dict(), f'{BASE_OUT}/unified_model_best.pt')
                with open(f'{BASE_OUT}/config.json', 'w') as f:
                    json.dump({"hdfs_threshold": val_metrics["threshold"], "vocab_size": len(vocab)}, f)
            else:
                patience_counter += 1
                
            if patience_counter >= CONFIG["mt_patience"]:
                break
                
        return best_mean_f1

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Optuna Weight Tuning main (Task 3.3)
# ─────────────────────────────────────────────────────────────────────────────
if 'multi_task_train_done' not in ckpt_state:
    print("\n" + "="*80)
    print("  [CELL 9] Running Multi-Task Joint Training & Optuna Optimization...")
    print("="*80)
    trainer = MultiTaskTrainer()
    
    if CONFIG["optuna_enabled"]:
        def objective(trial):
            w1 = trial.suggest_float("w1", 0.1, 1.0)
            w2 = trial.suggest_float("w2", 0.1, 1.0)
            w3 = trial.suggest_float("w3", 0.1, 1.0)
            s = w1 + w2 + w3
            return trainer.train_model(w1/s, w2/s, w3/s)
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=CONFIG["optuna_trials"])
        print(f"\nBest Optuna parameters: {study.best_params}")
        print(f"Best Val Mean F1: {study.best_value:.4f}")
    else:
        # equal weights
        trainer.train_model(1.0/3, 1.0/3, 1.0/3)
        
    ckpt_state['multi_task_train_done'] = True
    save_ckpt(ckpt_state)
    print("  Multi-Task Training complete!")
else:
    print("  [Step 9] Unified model weights found on disk.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9.5 — Final Evaluation on Test Partition
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("  [CELL 9.5] Final Evaluation on Test Partition...")
print("="*80)

best_model = UnifiedLogAnomalyModel(vocab_size=len(vocab), max_len=CONFIG["max_len"]).to(DEVICE)
best_model_path = f'{BASE_OUT}/unified_model_best.pt'
config_path = f'{BASE_OUT}/config.json'

if os.path.exists(best_model_path) and os.path.exists(config_path):
    best_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
    hdfs_threshold = saved_config["hdfs_threshold"]
    print(f"Loaded best unified model from {best_model_path}")
    print(f"HDFS optimal threshold: {hdfs_threshold:.6f}")
    
    # Load test datasets
    test_hdfs = TokenizedDataset(f'{PROCESSED_DIR}/hdfs_test.parquet')
    test_bgl = TokenizedDataset(f'{PROCESSED_DIR}/bgl_test.parquet')
    test_spirit = TokenizedDataset(f'{PROCESSED_DIR}/spirit_test.parquet')
    
    # Run evaluation
    results_df = evaluate_test_set(best_model, test_hdfs, test_bgl, test_spirit, hdfs_threshold, DEVICE)
    
    # Save results to CSV
    results_csv = f'{BASE_OUT}/unified_multitask_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved final test partition evaluation to {results_csv}")
    print("\n--- Final Test Set Metrics ---")
    print(results_df.to_string(index=False))
else:
    print("Warning: Best model weights or config not found. Skipping final evaluation.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — SHAP Explainability & Visualizations (Task 4)
# ─────────────────────────────────────────────────────────────────────────────
if 'explainability_done' not in ckpt_state:
    print("\n" + "="*80)
    print("  [CELL 10] Running SHAP Explainability and HDFS Heatmaps...")
    print("="*80)
    
    # Load model
    model = UnifiedLogAnomalyModel(vocab_size=len(vocab), max_len=CONFIG["max_len"]).to(DEVICE)
    model.load_state_dict(torch.load(f'{BASE_OUT}/unified_model_best.pt', map_location=DEVICE))
    model.eval()
    
    reverse_vocab = {v: k for k, v in vocab.items()}
    
    # 10.1 BGL / Spirit GradientExplainer
    class WrappedEmbeddingModel(nn.Module):
        def __init__(self, model, dataset_id):
            super().__init__()
            self.model = model
            self.dataset_id = dataset_id
        def forward(self, inputs_embeds):
            outputs = self.model.encoder.distilbert(inputs_embeds=inputs_embeds)
            token_embs = outputs.last_hidden_state
            if self.dataset_id == 1:
                return self.model.bgl_head(token_embs[:, 0, :]).unsqueeze(-1)
            else:
                return self.model.spirit_head(token_embs, None).unsqueeze(-1)

    # Disable CuDNN temporarily because it blocks PyTorch backprop in eval mode
    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    
    try:
        for name, (path, d_id) in [("bgl", (f'{PROCESSED_DIR}/bgl_test.parquet', 1)), 
                                   ("spirit", (f'{PROCESSED_DIR}/spirit_test.parquet', 2))]:
            ds = TokenizedDataset(path)
            loader = DataLoader(ds, batch_size=200, shuffle=False)
            batch = next(iter(loader))
            
            ids = batch["input_ids"].to(DEVICE)
            labels = batch["label"].cpu().numpy()
            
            # Find normal background & anomalous target examples
            norms = np.where(labels == 0)[0]
            anoms = np.where(labels == 1)[0]
            
            if len(norms) >= 20 and len(anoms) >= 2:
                bg_ids = ids[norms[:20]]
                test_ids = ids[anoms[:2]]
                
                with torch.no_grad():
                    bg_embeds = model.encoder.distilbert.embeddings(bg_ids)
                    test_embeds = model.encoder.distilbert.embeddings(test_ids)
                    
                wrapped = WrappedEmbeddingModel(model, d_id).to(DEVICE)
                explainer = shap.GradientExplainer(wrapped, bg_embeds)
                shap_values = explainer.shap_values(test_embeds)
                
                # shap_values shape [2, 128, 768] -> sum along embedding
                sg_vals = shap_values[0] if isinstance(shap_values, list) else shap_values
                step_importance = sg_vals.squeeze().sum(axis=-1)
                if len(step_importance.shape) == 1:
                    step_importance = np.expand_dims(step_importance, axis=0)
                    
                # Print attributions for first anomaly
                seq_ids = test_ids[0].cpu().numpy()
                seq_imp = step_importance[0]
                
                active_indices = [i for i, t_id in enumerate(seq_ids) if t_id != 0]
                valid_imp = seq_imp[active_indices]
                valid_ids = seq_ids[active_indices]
                
                top_pos = np.argsort(np.abs(valid_imp))[-5:][::-1]
                print(f"\n📊 Top SHAP attributions for anomalous {name} sequence:")
                for pos in top_pos:
                    t_id = valid_ids[pos]
                    template = reverse_vocab.get(t_id, "UNK")
                    print(f"  Pos {active_indices[pos]:03d} | SHAP: {valid_imp[pos]:+.4f} | '{template[:60]}'")
                    
                # Save plot
                plt.figure(figsize=(8, 4))
                plt.bar(range(len(valid_imp)), valid_imp, color=["#e53935" if v > 0 else "#1e88e5" for v in valid_imp])
                plt.title(f"SHAP Log Token Importance attributions — {name.upper()}", fontweight="bold")
                plt.tight_layout()
                plt.savefig(f'{BASE_OUT}/shap_attribution_{name}.png', dpi=200)
                plt.close()
    finally:
        torch.backends.cudnn.enabled = prev_cudnn
        
    # 10.2 HDFS Reconstruction error heatmap visualization
    hdfs_ds = TokenizedDataset(f'{PROCESSED_DIR}/hdfs_test.parquet')
    hdfs_loader = DataLoader(hdfs_ds, batch_size=200, shuffle=False)
    batch = next(iter(hdfs_loader))
    ids = batch["input_ids"].to(DEVICE)
    mask = batch["attention_mask"].to(DEVICE)
    labels = batch["label"].cpu().numpy()
    
    anoms = np.where(labels == 1)[0]
    target_idx = anoms[:5] if len(anoms) >= 5 else np.arange(5)
    
    t_ids, t_mask = ids[target_idx], mask[target_idx]
    d_ids = torch.zeros(t_ids.size(0), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        outputs = model(t_ids, t_mask, d_ids)
    pred, target = outputs["hdfs_reconstructed"], outputs["hdfs_embeddings"]
    
    active = (~t_mask).float().unsqueeze(-1)
    sq_err = ((pred - target) ** 2) * active
    seq_mse = sq_err.mean(dim=-1).cpu().numpy()
    
    heatmap_matrix = []
    y_labels = []
    for i in range(len(target_idx)):
        seq_len = np.sum(~t_mask[i].cpu().numpy())
        valid_errs = seq_mse[i][:seq_len]
        y_labels.append(f"Seq {target_idx[i]:03d}")
        
        row = np.zeros(CONFIG["max_len"])
        row[:seq_len] = valid_errs
        row[seq_len:] = np.nan
        heatmap_matrix.append(row)
        
    plt.figure(figsize=(10, 4))
    cmap = sns.color_palette("rocket", as_cmap=True)
    cmap.set_bad(color='#e0e0e0')
    sns.heatmap(np.array(heatmap_matrix), cmap=cmap, yticklabels=y_labels, xticklabels=10)
    plt.title("HDFS Session Reconstruction Error Heatmap (Grey = Masked Padding)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f'{BASE_OUT}/hdfs_reconstruction_heatmap.png', dpi=200)
    plt.close()
    print("\nSaved HDFS reconstruction heatmap to disk.")
    
    ckpt_state['explainability_done'] = True
    save_ckpt(ckpt_state)
    print("  Explainability complete!")
else:
    print("  [Step 10] Explainability plots found on disk.")

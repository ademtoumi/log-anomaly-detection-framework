
# ======================================================================
# # HierAttn-Block — Notebook 2: Baselines
# 
# **Requires:** `nb1_data_pipeline.ipynb` must have run first.
# 
# **Steps:** DeepLog → LogBERT
# 
# **Output:** `baseline_results.pkl`, `deeplog.pt`, `logbert.pt`
# ======================================================================

import os, random, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')


# ── Load cache from Notebook 1 ────────────────────────────────────────────────
CACHE_DIR = '/kaggle/working/hierattn_output/cache' if os.path.exists('/kaggle/working') else './results/hierattn_output/cache'
with open(os.path.join(CACHE_DIR, 'features.pkl'), 'rb') as f:
    C = pickle.load(f)

feat_train=C['feat_train']; feat_val=C['feat_val']; feat_test=C['feat_test']
y_train=C['y_train']; y_val=C['y_val']; y_test=C['y_test']
X_train_seq=C['X_train_seq']; X_val_seq=C['X_val_seq']; X_test_seq=C['X_test_seq']
VOCAB_SIZE=C['VOCAB_SIZE']; MAX_LEN=C['MAX_LEN']; BATCH_SIZE=C['BATCH_SIZE']
OUTPUT_DIR=C['OUTPUT_DIR']; FIGURE_DIR=C['FIGURE_DIR']; MODEL_DIR=C['MODEL_DIR']
CACHE_DIR=C['CACHE_DIR']
for d in [FIGURE_DIR, MODEL_DIR, CACHE_DIR]: os.makedirs(d, exist_ok=True)
print(f'Cache loaded. VOCAB_SIZE={VOCAB_SIZE}, Train={len(feat_train):,}, Val={len(feat_val):,}, Test={len(feat_test):,}')


# ── Shared HDFSDataset (needed for LogBERT dataloaders) ───────────────────────
class HDFSDataset(torch.utils.data.Dataset):
    def __init__(self, feat_list):
        self.data = feat_list
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        return (
            torch.tensor(d['event_ids'],      dtype=torch.long),
            torch.tensor(d['param_feats'],    dtype=torch.float32),
            torch.tensor(d['sin_time'],       dtype=torch.float32),
            torch.tensor(d['struct_feats'],   dtype=torch.float32),
            torch.tensor(d['attention_mask'], dtype=torch.float32),
            torch.tensor(d['label'],          dtype=torch.long),
            torch.tensor(d['repl_count'],     dtype=torch.float32),
        )

# Rebuild sampler
class_counts  = np.bincount(y_train)
class_weights = 1.0 / class_counts
sampler = WeightedRandomSampler(
    weights=torch.tensor(class_weights[y_train], dtype=torch.double),
    num_samples=len(feat_train), replacement=True)

baseline_results = {}
all_roc          = {}



# ======================================================================
# ## Step 5 — Baseline 1: DeepLog
# ======================================================================

class DeepLogLSTM(nn.Module):
    def __init__(self, vocab_size, hidden=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, hidden, num_layers, batch_first=True, dropout=0.3)
        self.head = nn.Linear(hidden, vocab_size)
    def forward(self, x):
        return self.head(self.lstm(self.embedding(x))[0])

class WindowDataset(Dataset):
    def __init__(self, seqs, window=10):
        self.pairs = []
        for seq in seqs:
            seq_np = seq if isinstance(seq, np.ndarray) else seq.numpy()
            nonzero = np.where(seq_np > 0)[0]
            if len(nonzero) < 2: continue
            clean = seq_np[nonzero]
            for i in range(len(clean) - 1):
                inp = clean[max(0, i-window+1):i+1]
                if len(inp) < window:
                    inp = np.pad(inp, (window - len(inp), 0), constant_values=0)
                self.pairs.append((inp, clean[i+1]))
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        inp, tgt = self.pairs[idx]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

DL_WINDOW = 10
DL_K      = 9
normal_mask = y_train == 0
dl_win_ds = WindowDataset(X_train_seq[normal_mask], window=DL_WINDOW)
dl_win_dl = DataLoader(dl_win_ds, batch_size=512, shuffle=True, num_workers=0)
print(f'Window pairs: {len(dl_win_ds):,}')


deeplog_model = DeepLogLSTM(VOCAB_SIZE).to(DEVICE)
dl_optim = torch.optim.Adam(deeplog_model.parameters(), lr=1e-3)
dl_crit  = nn.CrossEntropyLoss(ignore_index=0)

print('Training DeepLog (15 epochs, normal sessions only) ...')
for epoch in range(15):
    deeplog_model.train()
    total_loss = 0.0
    for inp_b, tgt_b in dl_win_dl:
        inp_b, tgt_b = inp_b.to(DEVICE), tgt_b.to(DEVICE)
        logits = deeplog_model(inp_b)[:, -1, :]
        loss   = dl_crit(logits, tgt_b)
        dl_optim.zero_grad(); loss.backward(); dl_optim.step()
        total_loss += loss.item()
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f'  Epoch {epoch+1:02d}/15 — Loss: {total_loss/len(dl_win_dl):.4f}')

del dl_win_ds, dl_win_dl
torch.cuda.empty_cache()


def deeplog_score_session(seq_np, window=DL_WINDOW, k=DL_K):
    nonzero = np.where(seq_np > 0)[0]
    if len(nonzero) < 2: return 0.0
    clean = seq_np[nonzero]
    anomalous, total = 0, 0
    for i in range(len(clean) - 1):
        inp = clean[max(0, i-window+1):i+1]
        if len(inp) < window:
            inp = np.pad(inp, (window - len(inp), 0), constant_values=0)
        x = torch.tensor(inp, dtype=torch.long).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            topk = deeplog_model(x)[:, -1, :].topk(k, dim=-1).indices.squeeze().cpu().numpy()
        if clean[i+1] not in topk: anomalous += 1
        total += 1
    return anomalous / total if total > 0 else 0.0

deeplog_model.eval()
print('Scoring validation set for threshold tuning ...')
val_dl_scores  = np.array([deeplog_score_session(X_val_seq[i])  for i in range(len(X_val_seq))])
print('Scoring test set ...')
dl_scores      = np.array([deeplog_score_session(X_test_seq[i]) for i in range(len(X_test_seq))])

best_dl_thresh, best_dl_f1 = 0.0, 0.0
for thr in np.linspace(0, 1, 101):
    f1 = f1_score(y_val, (val_dl_scores >= thr).astype(int), zero_division=0)
    if f1 > best_dl_f1: best_dl_f1, best_dl_thresh = f1, thr

dl_preds = (dl_scores >= best_dl_thresh).astype(int)
dl_prec  = precision_score(y_test, dl_preds, zero_division=0)
dl_rec   = recall_score(y_test, dl_preds,    zero_division=0)
dl_f1    = f1_score(y_test, dl_preds,        zero_division=0)
try:    dl_auc = roc_auc_score(y_test, dl_scores)
except: dl_auc = 0.0

baseline_results['DeepLog'] = {'Precision': round(dl_prec,4), 'Recall': round(dl_rec,4), 'F1': round(dl_f1,4), 'AUC': round(dl_auc,4)}
print(f'DeepLog → P={dl_prec:.4f}  R={dl_rec:.4f}  F1={dl_f1:.4f}  AUC={dl_auc:.4f}')
try:
    dl_fpr, dl_tpr, _ = roc_curve(y_test, dl_scores)
    all_roc['DeepLog'] = (dl_fpr, dl_tpr, dl_auc)
except: pass


# Confusion matrix DeepLog
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, dl_preds), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal','Anomaly'], yticklabels=['Normal','Anomaly'], ax=ax)
ax.set_title('DeepLog — Confusion Matrix', fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'cm_deeplog.png'), dpi=300)
plt.close(fig)
torch.save(deeplog_model.state_dict(), os.path.join(MODEL_DIR, 'deeplog.pt'))
print('Saved: cm_deeplog.png + deeplog.pt')



# ======================================================================
# ## Step 6 — Baseline 2: LogBERT
# ======================================================================

class LogBERTModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, max_len=MAX_LEN+1, dropout=0.1):
        super().__init__()
        self.CLS_ID     = vocab_size
        self.vocab_size = vocab_size + 1
        self.token_emb  = nn.Embedding(self.vocab_size, d_model, padding_idx=0)
        self.pos_emb    = nn.Embedding(max_len + 1, d_model)
        self.norm_in    = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256,
                                               dropout=dropout, batch_first=True, norm_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.mlm_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                      nn.LayerNorm(d_model), nn.Linear(d_model, self.vocab_size))
        self.cls_head = nn.Linear(d_model, 2)

    def forward(self, input_ids, attention_mask=None, return_cls=False):
        B, L = input_ids.shape
        pos  = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x    = self.dropout(self.norm_in(self.token_emb(input_ids) + self.pos_emb(pos)))
        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        if return_cls: return self.cls_head(x[:, 0, :])
        return self.mlm_head(x[:, 1:, :])

class LogBERTDataset(Dataset):
    def __init__(self, feat_list, mask_rate=0.15, mode='pretrain', vocab_size=VOCAB_SIZE):
        self.data=feat_list; self.mask_rate=mask_rate; self.mode=mode
        self.vocab_size=vocab_size; self.CLS_ID=vocab_size
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d   = self.data[idx]
        evs = d['event_ids'].copy(); msk = d['attention_mask'].copy()
        cls_arr  = np.array([self.CLS_ID], dtype=np.int64)
        input_ids = np.concatenate([cls_arr, evs])
        attn_mask = np.concatenate([np.array([1.0], dtype=np.float32), msk])
        if self.mode == 'pretrain':
            masked_ids = input_ids.copy(); mlm_labels = input_ids.copy()
            for i in range(1, len(masked_ids)):
                if attn_mask[i] == 1.0 and random.random() < self.mask_rate:
                    r = random.random()
                    if r < 0.8:   masked_ids[i] = self.vocab_size
                    elif r < 0.9: masked_ids[i] = random.randint(2, self.vocab_size - 1)
                else: mlm_labels[i] = -100
            return (torch.tensor(masked_ids, dtype=torch.long),
                    torch.tensor(attn_mask,  dtype=torch.float32),
                    torch.tensor(mlm_labels[1:], dtype=torch.long))
        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(attn_mask, dtype=torch.float32),
                torch.tensor(d['label'], dtype=torch.long))


lb_model    = LogBERTModel(VOCAB_SIZE).to(DEVICE)
lb_optim_pre = AdamW(lb_model.parameters(), lr=1e-3, weight_decay=1e-4)

normal_feats  = [f for f in feat_train if f['label'] == 0]
lb_pre_ds     = LogBERTDataset(normal_feats, mask_rate=0.15, mode='pretrain')
lb_pre_dl     = DataLoader(lb_pre_ds, batch_size=128, shuffle=True, num_workers=0)

print('Pre-training LogBERT (MLM, 10 epochs, normal sessions only) ...')
for epoch in range(10):
    lb_model.train(); total_loss = 0.0
    for masked_ids, attn_mask, mlm_labels in lb_pre_dl:
        masked_ids=masked_ids.to(DEVICE); attn_mask=attn_mask.to(DEVICE); mlm_labels=mlm_labels.to(DEVICE)
        logits = lb_model(masked_ids, attention_mask=attn_mask, return_cls=False)
        loss   = F.cross_entropy(logits.reshape(-1, lb_model.vocab_size), mlm_labels.reshape(-1), ignore_index=-100)
        lb_optim_pre.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(lb_model.parameters(), 1.0)
        lb_optim_pre.step(); total_loss += loss.item()
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f'  Pre-train Epoch {epoch+1:02d}/10 — MLM Loss: {total_loss/len(lb_pre_dl):.4f}')

del lb_pre_ds, lb_pre_dl; torch.cuda.empty_cache()


# Fine-tune with separate sampler
lb_sample_weights = class_weights[y_train]
lb_sampler = WeightedRandomSampler(torch.tensor(lb_sample_weights, dtype=torch.double), len(feat_train), True)

lb_ft_train = LogBERTDataset(feat_train, mode='finetune')
lb_ft_val   = LogBERTDataset(feat_val,   mode='finetune')
lb_ft_test  = LogBERTDataset(feat_test,  mode='finetune')
lb_ft_dl    = DataLoader(lb_ft_train, batch_size=128, sampler=lb_sampler, num_workers=0)
lb_val_dl   = DataLoader(lb_ft_val,   batch_size=128, shuffle=False,      num_workers=0)
lb_test_dl  = DataLoader(lb_ft_test,  batch_size=128, shuffle=False,      num_workers=0)

n_neg = (y_train==0).sum(); n_pos = (y_train==1).sum()
lb_cls_w  = torch.tensor([1.0, n_neg/max(n_pos,1)], dtype=torch.float32).to(DEVICE)
lb_crit   = nn.CrossEntropyLoss(weight=lb_cls_w)
lb_opt_ft = AdamW(lb_model.parameters(), lr=5e-4, weight_decay=1e-4)

best_lb_f1, best_lb_state = 0.0, None
print('Fine-tuning LogBERT (15 epochs) ...')
for epoch in range(15):
    lb_model.train()
    for inp_ids, attn_mask, labels in lb_ft_dl:
        inp_ids=inp_ids.to(DEVICE); attn_mask=attn_mask.to(DEVICE); labels=labels.to(DEVICE)
        loss = lb_crit(lb_model(inp_ids, attention_mask=attn_mask, return_cls=True), labels)
        lb_opt_ft.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(lb_model.parameters(), 1.0)
        lb_opt_ft.step()
    lb_model.eval()
    vp, vpr = [], []
    with torch.no_grad():
        for inp_ids, attn_mask, labels in lb_val_dl:
            probs = F.softmax(lb_model(inp_ids.to(DEVICE), attention_mask=attn_mask.to(DEVICE), return_cls=True), dim=-1)[:,1].cpu().numpy()
            vp.extend((probs>=0.5).astype(int).tolist()); vpr.extend(probs.tolist())
    vf1 = f1_score(y_val, vp, zero_division=0)
    if vf1 > best_lb_f1:
        best_lb_f1=vf1; best_lb_state={k: v.cpu().clone() for k,v in lb_model.state_dict().items()}
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f'  Fine-tune Epoch {epoch+1:02d}/15 — Val F1: {vf1:.4f}  (best: {best_lb_f1:.4f})')

lb_model.load_state_dict(best_lb_state); lb_model = lb_model.to(DEVICE)
del lb_ft_dl; torch.cuda.empty_cache()


lb_model.eval()
lb_preds_test, lb_probs_test = [], []
with torch.no_grad():
    for inp_ids, attn_mask, labels in lb_test_dl:
        probs = F.softmax(lb_model(inp_ids.to(DEVICE), attention_mask=attn_mask.to(DEVICE), return_cls=True), dim=-1)[:,1].cpu().numpy()
        lb_preds_test.extend((probs>=0.5).astype(int).tolist()); lb_probs_test.extend(probs.tolist())
lb_preds_test=np.array(lb_preds_test); lb_probs_test=np.array(lb_probs_test)

lb_prec=precision_score(y_test,lb_preds_test,zero_division=0)
lb_rec =recall_score(y_test,lb_preds_test,zero_division=0)
lb_f1  =f1_score(y_test,lb_preds_test,zero_division=0)
try:    lb_auc=roc_auc_score(y_test,lb_probs_test)
except: lb_auc=0.0
baseline_results['LogBERT']={'Precision':round(lb_prec,4),'Recall':round(lb_rec,4),'F1':round(lb_f1,4),'AUC':round(lb_auc,4)}
print(f'LogBERT → P={lb_prec:.4f}  R={lb_rec:.4f}  F1={lb_f1:.4f}  AUC={lb_auc:.4f}')
try:
    lb_fpr,lb_tpr,_=roc_curve(y_test,lb_probs_test); all_roc['LogBERT']=(lb_fpr,lb_tpr,lb_auc)
except: pass

fig,ax=plt.subplots(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test,lb_preds_test),annot=True,fmt='d',cmap='Oranges',
            xticklabels=['Normal','Anomaly'],yticklabels=['Normal','Anomaly'],ax=ax)
ax.set_title('LogBERT — Confusion Matrix',fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout(); fig.savefig(os.path.join(FIGURE_DIR,'cm_logbert.png'),dpi=300); plt.close(fig)
torch.save(lb_model.state_dict(), os.path.join(MODEL_DIR,'logbert.pt'))
print('Saved: cm_logbert.png + logbert.pt')


# ── Save baseline cache ───────────────────────────────────────────────────────
with open(os.path.join(CACHE_DIR,'baseline_results.pkl'),'wb') as f:
    pickle.dump({'baseline_results':baseline_results,'all_roc':all_roc,
                 'dl_scores':dl_scores,'lb_probs_test':lb_probs_test,'y_test':y_test}, f)
print('\n✅ baseline_results.pkl saved')
print('\nBaseline Summary:')
for m,v in baseline_results.items():
    print(f'  {m:10s} → P={v["Precision"]}  R={v["Recall"]}  F1={v["F1"]}  AUC={v["AUC"]}')
print('\n✅ Notebook 2 complete — run nb3_hierattn_training.ipynb next')


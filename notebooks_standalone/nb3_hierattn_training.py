
# ======================================================================
# # HierAttn-Block — Notebook 3: Model Training & Inference
# 
# **Requires:** nb1 + nb2 must have run.
# 
# **Steps:** Architecture → Training → Two-Stage Inference
# 
# **Output:** `hierattn_best.pt`, `hierattn_results.pkl`
# ======================================================================

import os, random, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')


# ── Load caches ───────────────────────────────────────────────────────────────
CACHE_DIR = '/kaggle/working/hierattn_output/cache' if os.path.exists('/kaggle/working') else './results/hierattn_output/cache'
with open(os.path.join(CACHE_DIR,'features.pkl'),'rb') as f: C = pickle.load(f)
with open(os.path.join(CACHE_DIR,'baseline_results.pkl'),'rb') as f: B = pickle.load(f)

feat_train=C['feat_train']; feat_val=C['feat_val']; feat_test=C['feat_test']
y_train=C['y_train']; y_val=C['y_val']; y_test=C['y_test']
VOCAB_SIZE=C['VOCAB_SIZE']; MAX_LEN=C['MAX_LEN']; BATCH_SIZE=C['BATCH_SIZE']
MAX_EPOCHS=C['MAX_EPOCHS']; LR=C['LR']; WEIGHT_DECAY=C['WEIGHT_DECAY']; PATIENCE=C['PATIENCE']
OUTPUT_DIR=C['OUTPUT_DIR']; CACHE_DIR=C['CACHE_DIR']; MODEL_DIR=C['MODEL_DIR']; FIGURE_DIR=C['FIGURE_DIR']
test_missing_alloc=C['test_missing_alloc']; test_repl_neq3=C['test_repl_neq3']
val_missing_alloc=C['val_missing_alloc'];   val_repl_neq3=C['val_repl_neq3']
baseline_results=B['baseline_results']; all_roc=B['all_roc']
for d in [MODEL_DIR, CACHE_DIR, FIGURE_DIR]: os.makedirs(d, exist_ok=True)
print(f'Caches loaded. VOCAB={VOCAB_SIZE}, Train={len(feat_train):,}')


class HDFSDataset(Dataset):
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

class_counts  = np.bincount(y_train)
class_weights = 1.0 / class_counts
sampler = WeightedRandomSampler(
    weights=torch.tensor(class_weights[y_train], dtype=torch.double),
    num_samples=len(feat_train), replacement=True)

dl_train = DataLoader(HDFSDataset(feat_train), batch_size=BATCH_SIZE, sampler=sampler,  num_workers=0, pin_memory=False)
dl_val   = DataLoader(HDFSDataset(feat_val),   batch_size=BATCH_SIZE, shuffle=False,    num_workers=0)
dl_test  = DataLoader(HDFSDataset(feat_test),  batch_size=BATCH_SIZE, shuffle=False,    num_workers=0)
print(f'DataLoaders ready. Batches/epoch: {len(dl_train)}')



# ======================================================================
# ## Step 7 — HierAttn-Block Architecture
# ======================================================================

class EventEmbedding(nn.Module):
    """Template ID + param feats + sinusoidal time → 128-dim per event."""
    def __init__(self, vocab_size, embed_dim=64, param_dim=32, time_dim=32):
        super().__init__()
        self.template_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.param_proj   = nn.Linear(3, param_dim)
        assert embed_dim + param_dim + time_dim == 128
    def forward(self, event_ids, param_feats, sin_time):
        return torch.cat([self.template_emb(event_ids), self.param_proj(param_feats), sin_time], dim=-1)

class TransformerEncoder(nn.Module):
    """2-layer Transformer. Session vec = concat(mean_pool, max_pool) → 256-dim."""
    def __init__(self, d_model=128, nhead=4, num_layers=2, ffn_dim=256, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=ffn_dim, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
    def forward(self, x, attention_mask):
        pad_mask = (attention_mask == 0)
        H        = self.encoder(x, src_key_padding_mask=pad_mask)
        mask_exp = attention_mask.unsqueeze(-1)
        H_masked = H * mask_exp
        mean_vec = H_masked.sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        max_vec  = (H_masked + (1 - mask_exp) * (-1e9)).max(dim=1).values
        return H, torch.cat([mean_vec, max_vec], dim=-1)

class StructuralMLP(nn.Module):
    """11-dim structural features → 64-dim."""
    def __init__(self, in_dim=11, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,hidden), nn.ReLU(), nn.Linear(hidden,hidden), nn.ReLU())
    def forward(self, x): return self.net(x)

class HierAttnBlock(nn.Module):
    """Full dual-path model: Transformer + Structural MLP → fused classifier."""
    def __init__(self, vocab_size, embed_dim=64, param_dim=32, time_dim=32,
                 d_model=128, nhead=4, num_enc_layers=2, ffn_dim=256,
                 struct_dim=11, struct_hidden=64, fusion_hidden=128, dropout=0.3):
        super().__init__()
        self.event_emb   = EventEmbedding(vocab_size, embed_dim, param_dim, time_dim)
        self.transformer = TransformerEncoder(d_model, nhead, num_enc_layers, ffn_dim)
        self.struct_mlp  = StructuralMLP(struct_dim, struct_hidden)
        fuse_in = d_model * 2 + struct_hidden
        self.fusion   = nn.Sequential(nn.Linear(fuse_in, fusion_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.cls_head = nn.Linear(fusion_hidden, 2)
        self.aux_head = nn.Linear(fusion_hidden, 1)

    def forward(self, event_ids, param_feats, sin_time, struct_feats, attention_mask, return_aux=True):
        x             = self.event_emb(event_ids, param_feats, sin_time)
        H, sess_vec   = self.transformer(x, attention_mask)
        s_vec         = self.struct_mlp(struct_feats)
        hidden        = self.fusion(torch.cat([sess_vec, s_vec], dim=-1))
        logits        = self.cls_head(hidden)
        aux_out       = self.aux_head(hidden) if return_aux else None
        return logits, aux_out, H

model_hier  = HierAttnBlock(VOCAB_SIZE).to(DEVICE)
total_params = sum(p.numel() for p in model_hier.parameters() if p.requires_grad)
print(f'HierAttn-Block parameters: {total_params:,}')



# ======================================================================
# ## Step 8 — Training
# ======================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha
    def forward(self, logits, targets):
        pt      = F.softmax(logits, dim=-1)[range(len(targets)), targets]
        alpha_t = torch.where(targets==1,
            torch.tensor(self.alpha, device=logits.device),
            torch.tensor(1-self.alpha, device=logits.device))
        return (-alpha_t * (1-pt)**self.gamma * torch.log(pt+1e-8)).mean()

focal_loss = FocalLoss(gamma=2.0, alpha=0.75)
aux_loss   = nn.MSELoss()
optimizer  = AdamW(model_hier.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler  = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

history = {'train_loss':[], 'train_f1':[], 'val_loss':[], 'val_f1':[]}
best_val_f1, patience_count, best_state = 0.0, 0, None
CHECKPOINT = os.path.join(MODEL_DIR, 'hierattn_best.pt')

print(f'Training HierAttn-Block: max {MAX_EPOCHS} epochs, patience={PATIENCE}')
print(f'Loss: FocalLoss(γ=2, α=0.75) + 0.1·MSE(replication count)')

for epoch in range(MAX_EPOCHS):
    model_hier.train()
    ep_loss, ep_preds, ep_labels = 0.0, [], []
    for batch in dl_train:
        ev,pf,st,sf,am,labels,repl = [b.to(DEVICE) for b in batch]
        logits, aux_out, _ = model_hier(ev,pf,st,sf,am,return_aux=True)
        loss = focal_loss(logits,labels) + 0.1*aux_loss(aux_out.squeeze(-1),repl)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model_hier.parameters(), 1.0)
        optimizer.step()
        ep_loss += loss.item()
        ep_preds.extend(logits.argmax(-1).cpu().numpy().tolist())
        ep_labels.extend(labels.cpu().numpy().tolist())
    scheduler.step()
    train_f1   = f1_score(ep_labels, ep_preds, zero_division=0)
    train_loss = ep_loss / len(dl_train)

    model_hier.eval()
    vl, vp, vl_true = 0.0, [], []
    with torch.no_grad():
        for batch in dl_val:
            ev,pf,st,sf,am,labels,repl = [b.to(DEVICE) for b in batch]
            logits, aux_out, _ = model_hier(ev,pf,st,sf,am,return_aux=True)
            vl += (focal_loss(logits,labels) + 0.1*aux_loss(aux_out.squeeze(-1),repl)).item()
            vp.extend(logits.argmax(-1).cpu().numpy().tolist())
            vl_true.extend(labels.cpu().numpy().tolist())
    val_f1  = f1_score(vl_true, vp, zero_division=0)
    val_loss = vl / len(dl_val)

    history['train_loss'].append(train_loss); history['train_f1'].append(train_f1)
    history['val_loss'].append(val_loss);     history['val_f1'].append(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1=val_f1; patience_count=0
        best_state={k: v.cpu().clone() for k,v in model_hier.state_dict().items()}
        torch.save(best_state, CHECKPOINT)
    else:
        patience_count += 1

    if (epoch+1)%5==0 or epoch==0 or patience_count==0:
        print(f'  Ep {epoch+1:02d}/{MAX_EPOCHS} | '
              f'TrLoss:{train_loss:.4f} TrF1:{train_f1:.4f} | '
              f'VaLoss:{val_loss:.4f} VaF1:{val_f1:.4f} | '
              f'Best:{best_val_f1:.4f} Pat:{patience_count}/{PATIENCE}')

    if patience_count >= PATIENCE:
        print(f'  Early stopping at epoch {epoch+1}')
        break

print(f'\nTraining complete. Best Val F1: {best_val_f1:.4f}')



# ======================================================================
# ## Step 9 — Two-Stage Inference
# ======================================================================

model_hier.load_state_dict(best_state); model_hier = model_hier.to(DEVICE); model_hier.eval()

# Tune threshold τ on validation set
val_probs_hier = []
with torch.no_grad():
    for batch in dl_val:
        ev,pf,st,sf,am,labels,repl = [b.to(DEVICE) for b in batch]
        probs = F.softmax(model_hier(ev,pf,st,sf,am,return_aux=False)[0], dim=-1)[:,1].cpu().numpy()
        val_probs_hier.extend(probs.tolist())
val_probs_hier = np.array(val_probs_hier)

stage1_val = (val_missing_alloc==1) | (val_repl_neq3==1)
best_tau, best_2stage_f1 = 0.5, 0.0
for tau in np.linspace(0.1, 0.9, 81):
    combined = np.maximum(stage1_val.astype(int), (val_probs_hier >= tau).astype(int))
    f1 = f1_score(y_val, combined, zero_division=0)
    if f1 > best_2stage_f1: best_2stage_f1, best_tau = f1, tau
print(f'Best τ = {best_tau:.2f}  (Val 2-stage F1: {best_2stage_f1:.4f})')

# Stage 1: hard rules on test
stage1_test = (test_missing_alloc==1) | (test_repl_neq3==1)
print(f'Stage 1: {stage1_test.sum()} sessions flagged ({100*stage1_test.sum()/len(y_test):.1f}% of test)')

# Stage 2: neural model
hier_probs_test = []
with torch.no_grad():
    for batch in dl_test:
        ev,pf,st,sf,am,labels,repl = [b.to(DEVICE) for b in batch]
        probs = F.softmax(model_hier(ev,pf,st,sf,am,return_aux=False)[0], dim=-1)[:,1].cpu().numpy()
        hier_probs_test.extend(probs.tolist())
hier_probs_test  = np.array(hier_probs_test)
hier_final_preds = np.maximum(stage1_test.astype(int), (hier_probs_test>=best_tau).astype(int))
hier_final_probs = np.where(stage1_test, 1.0, hier_probs_test)

hier_prec = precision_score(y_test, hier_final_preds, zero_division=0)
hier_rec  = recall_score(y_test, hier_final_preds,    zero_division=0)
hier_f1   = f1_score(y_test, hier_final_preds,        zero_division=0)
try:    hier_auc = roc_auc_score(y_test, hier_final_probs)
except: hier_auc = 0.0

results = {'HierAttnBlock': {'Precision':round(hier_prec,4),'Recall':round(hier_rec,4),
                              'F1':round(hier_f1,4),'AUC':round(hier_auc,4)}}
print(f'\nHierAttn-Block → P={hier_prec:.4f}  R={hier_rec:.4f}  F1={hier_f1:.4f}  AUC={hier_auc:.4f}')

try:
    h_fpr,h_tpr,_ = roc_curve(y_test, hier_final_probs)
    all_roc['HierAttn-Block'] = (h_fpr, h_tpr, hier_auc)
except: pass


# ── Save results cache ────────────────────────────────────────────────────────
with open(os.path.join(CACHE_DIR,'hierattn_results.pkl'),'wb') as f:
    pickle.dump({
        'results':results, 'all_roc':all_roc,
        'hier_final_preds':hier_final_preds, 'hier_final_probs':hier_final_probs,
        'hier_prec':hier_prec, 'hier_rec':hier_rec, 'hier_f1':hier_f1, 'hier_auc':hier_auc,
        'history':history,
        'model_class': 'HierAttnBlock',
        'VOCAB_SIZE': VOCAB_SIZE,
    }, f)

del dl_train, dl_val, dl_test; torch.cuda.empty_cache()
print('\n✅ hierattn_results.pkl saved')
print('✅ Notebook 3 complete — run nb4_evaluation_figures.ipynb next')


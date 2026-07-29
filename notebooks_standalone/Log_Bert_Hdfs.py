# =============================================================================
# NOTEBOOK 4 — ABLATION STUDY + THESIS FIGURES + FINAL RESULTS
# =============================================================================
# Input datasets (attach all 3 in Kaggle Add Data):
#   hierattn_output   → features.pkl
#   hierattn_output1  → baseline_results.pkl, deeplog.pt, logbert.pt
#   hierattn_output2  → hierattn_results.pkl, hierattn_best.pt
# =============================================================================

import os, random, pickle, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
})
PALETTE = ['#2E86AB', '#E84855', '#3BB273', '#F18F01', '#A23B72', '#C73E1D']

print('='*65)
print('  NOTEBOOK 4 — Ablation + Figures + Final Results')
print(f'  Device: {DEVICE}')
print('='*65)


# =============================================================================
# LOAD ALL CACHES — recursive search handles any Kaggle dataset naming
# =============================================================================

def find_pkl(filename):
    """Recursively search /kaggle/input for a file by name."""
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if filename in files:
                return os.path.join(root, filename)
    # local fallback
    local = f'./results/hierattn_output/cache/{filename}'
    if os.path.exists(local):
        return local
    raise FileNotFoundError(f'{filename} not found under /kaggle/input')

features_path  = find_pkl('features.pkl')
baselines_path = find_pkl('baseline_results.pkl')
hierattn_path  = find_pkl('hierattn_results.pkl')
print(f'features.pkl       : {features_path}')
print(f'baseline_results   : {baselines_path}')
print(f'hierattn_results   : {hierattn_path}')

with open(features_path,  'rb') as f: C = pickle.load(f)
with open(baselines_path, 'rb') as f: B = pickle.load(f)
with open(hierattn_path,  'rb') as f: H = pickle.load(f)

# ── Unpack features cache ─────────────────────────────────────────────────────
feat_train      = C['feat_train'];      feat_val   = C['feat_val'];   feat_test  = C['feat_test']
y_train         = C['y_train'];         y_val      = C['y_val'];      y_test     = C['y_test']
X_train_struct  = C['X_train_struct'];  X_val_struct = C['X_val_struct']
X_test_struct   = C['X_test_struct']
VOCAB_SIZE      = C['VOCAB_SIZE'];      MAX_LEN    = C['MAX_LEN'];    BATCH_SIZE = C['BATCH_SIZE']
MAX_EPOCHS      = C['MAX_EPOCHS'];      LR         = C['LR']
WEIGHT_DECAY    = C['WEIGHT_DECAY'];    PATIENCE   = C['PATIENCE']
print(f'  features.pkl loaded — VOCAB={VOCAB_SIZE}, Train={len(feat_train):,}')

# ── Unpack baselines cache ────────────────────────────────────────────────────
baseline_results = B['baseline_results']
all_roc          = B['all_roc']
dl_scores        = B['dl_scores']
lb_probs_test    = B['lb_probs_test']
print(f'  baseline_results.pkl loaded — {list(baseline_results.keys())}')

# ── Unpack hierattn cache ─────────────────────────────────────────────────────
results          = H['results'];           history         = H['history']
hier_final_preds = H['hier_final_preds'];  hier_final_probs = H['hier_final_probs']
hier_prec        = H['hier_prec'];         hier_rec        = H['hier_rec']
hier_f1          = H['hier_f1'];           hier_auc        = H['hier_auc']
print(f'  hierattn_results.pkl loaded — HierAttn F1={hier_f1:.4f}')

# ── Output directories ────────────────────────────────────────────────────────
OUTPUT_DIR = '/kaggle/working/hierattn_output'
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
MODEL_DIR  = os.path.join(OUTPUT_DIR, 'models')
for d in [OUTPUT_DIR, FIGURE_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# Reconstruct baseline preds from stored scores/probs
dl_preds_final = (dl_scores        >= 0.5).astype(int)
lb_preds_final = (lb_probs_test    >= 0.5).astype(int)


# =============================================================================
# MODEL CLASSES  (must match nb3 exactly)
# =============================================================================

class HDFSDataset(Dataset):
    def __init__(self, feat_list): self.data = feat_list
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

class EventEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, param_dim=32):
        super().__init__()
        self.template_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.param_proj   = nn.Linear(3, param_dim)
    def forward(self, ev, pf, st):
        return torch.cat([self.template_emb(ev), self.param_proj(pf), st], dim=-1)

class SessionTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, ffn_dim=256, dropout=0.1):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=ffn_dim, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
    def forward(self, x, am):
        H   = self.encoder(x, src_key_padding_mask=(am == 0))
        m   = am.unsqueeze(-1)
        Hm  = H * m
        avg = Hm.sum(1) / m.sum(1).clamp(min=1)
        mx  = (Hm + (1 - m) * (-1e9)).max(1).values
        return H, torch.cat([avg, mx], dim=-1)

class StructuralMLP(nn.Module):
    def __init__(self, in_dim=11, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
    def forward(self, x): return self.net(x)

class HierAttnBlock(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, param_dim=32,
                 d_model=128, nhead=4, num_enc_layers=2, ffn_dim=256,
                 struct_dim=11, struct_hidden=64, fusion_hidden=128, dropout=0.3):
        super().__init__()
        self.event_emb   = EventEmbedding(vocab_size, embed_dim, param_dim)
        self.transformer = SessionTransformer(d_model, nhead, num_enc_layers, ffn_dim)
        self.struct_mlp  = StructuralMLP(struct_dim, struct_hidden)
        self.fusion   = nn.Sequential(
            nn.Linear(d_model * 2 + struct_hidden, fusion_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.cls_head = nn.Linear(fusion_hidden, 2)
        self.aux_head = nn.Linear(fusion_hidden, 1)

    def forward(self, ev, pf, st, sf, am, return_aux=True):
        x           = self.event_emb(ev, pf, st)
        H, sess_vec = self.transformer(x, am)
        s_vec       = self.struct_mlp(sf)
        hidden      = self.fusion(torch.cat([sess_vec, s_vec], dim=-1))
        return self.cls_head(hidden), (self.aux_head(hidden) if return_aux else None), H

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__(); self.gamma = gamma; self.alpha = alpha
    def forward(self, logits, targets):
        pt = F.softmax(logits, -1)[range(len(targets)), targets]
        at = torch.where(targets == 1,
            torch.tensor(self.alpha,   device=logits.device),
            torch.tensor(1-self.alpha, device=logits.device))
        return (-at * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)).mean()

# ── Shared training utilities ─────────────────────────────────────────────────
class_counts  = np.bincount(y_train)
class_weights = 1.0 / class_counts
focal_crit    = FocalLoss(2.0, 0.75)
aux_crit      = nn.MSELoss()

def make_loaders():
    s = WeightedRandomSampler(
        torch.tensor(class_weights[y_train], dtype=torch.double), len(feat_train), True)
    return (DataLoader(HDFSDataset(feat_train), BATCH_SIZE, sampler=s,     num_workers=0),
            DataLoader(HDFSDataset(feat_val),   BATCH_SIZE, shuffle=False, num_workers=0),
            DataLoader(HDFSDataset(feat_test),  BATCH_SIZE, shuffle=False, num_workers=0))

print('Model classes ready.')


# =============================================================================
# STEP 11 — ABLATION STUDY
# =============================================================================
print('\nSTEP 11 — ABLATION STUDY')
print('-'*40)

def train_variant(model, use_aux=True):
    dl_tr, dl_v, _ = make_loaders()
    opt   = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
    best_f1, best_st, pat = 0.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in dl_tr:
            ev, pf, st, sf, am, labels, repl = [b.to(DEVICE) for b in batch]
            logits, aux_out, _ = model(ev, pf, st, sf, am, return_aux=use_aux)
            loss = focal_crit(logits, labels)
            if use_aux and aux_out is not None:
                loss = loss + 0.1 * aux_crit(aux_out.squeeze(-1), repl)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval(); preds = []
        with torch.no_grad():
            for batch in dl_v:
                ev, pf, st, sf, am, labels, repl = [b.to(DEVICE) for b in batch]
                preds.extend(
                    model(ev, pf, st, sf, am, return_aux=False)[0]
                    .argmax(-1).cpu().numpy())
        vf1 = f1_score(y_val, preds, zero_division=0)
        if vf1 > best_f1:
            best_f1 = vf1
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= PATIENCE:
            break
    if best_st:
        model.load_state_dict(best_st)
    return model

def eval_variant(model):
    _, _, dl_tst = make_loaders()
    model.eval(); preds, probs = [], []
    with torch.no_grad():
        for batch in dl_tst:
            ev, pf, st, sf, am, labels, repl = [b.to(DEVICE) for b in batch]
            p = F.softmax(model(ev, pf, st, sf, am, return_aux=False)[0], -1)[:,1].cpu().numpy()
            probs.extend(p.tolist())
            preds.extend((p >= 0.5).astype(int).tolist())
    preds = np.array(preds); probs = np.array(probs)
    try:    auc = roc_auc_score(y_test, probs)
    except: auc = 0.0
    return preds, probs, {
        'Precision': round(precision_score(y_test, preds, zero_division=0), 4),
        'Recall':    round(recall_score(y_test,    preds, zero_division=0), 4),
        'F1':        round(f1_score(y_test,        preds, zero_division=0), 4),
        'AUC':       round(auc, 4),
    }

# ── Variant 1: Sequence Only (no structural MLP) ──────────────────────────────
class HierAttnSeqOnly(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.event_emb   = EventEmbedding(vocab_size)
        self.transformer = SessionTransformer()
        self.fusion      = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3))
        self.cls_head    = nn.Linear(128, 2)
        self.aux_head    = nn.Linear(128, 1)
    def forward(self, ev, pf, st, sf, am, return_aux=True):
        _, sv = self.transformer(self.event_emb(ev, pf, st), am)
        h = self.fusion(sv)
        return self.cls_head(h), (self.aux_head(h) if return_aux else None), torch.zeros(1)

# ── Variant 2: Structural Only (no transformer) ───────────────────────────────
class HierAttnStructOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.struct_mlp = StructuralMLP(11, 64)
        self.fusion     = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3))
        self.cls_head   = nn.Linear(128, 2)
        self.aux_head   = nn.Linear(128, 1)
    def forward(self, ev, pf, st, sf, am, return_aux=True):
        h = self.fusion(self.struct_mlp(sf))
        return self.cls_head(h), (self.aux_head(h) if return_aux else None), torch.zeros(1)

# ── Run ablations ─────────────────────────────────────────────────────────────
print('  [1/4] Sequence Only ...')
m1 = train_variant(HierAttnSeqOnly(VOCAB_SIZE).to(DEVICE), use_aux=True)
_, probs_seq, res_seq = eval_variant(m1)
print(f'        {res_seq}')
try:
    f_, t_, _ = roc_curve(y_test, probs_seq)
    all_roc['Seq Only'] = (f_, t_, res_seq['AUC'])
except: pass
del m1; torch.cuda.empty_cache()

print('  [2/4] Structural Only ...')
m2 = train_variant(HierAttnStructOnly().to(DEVICE), use_aux=True)
_, probs_struct, res_struct = eval_variant(m2)
print(f'        {res_struct}')
try:
    f_, t_, _ = roc_curve(y_test, probs_struct)
    all_roc['Struct Only'] = (f_, t_, res_struct['AUC'])
except: pass
del m2; torch.cuda.empty_cache()

print('  [3/4] No Auxiliary Head ...')
m3 = train_variant(HierAttnBlock(VOCAB_SIZE).to(DEVICE), use_aux=False)
_, probs_noaux, res_noaux = eval_variant(m3)
print(f'        {res_noaux}')
try:
    f_, t_, _ = roc_curve(y_test, probs_noaux)
    all_roc['No Aux Head'] = (f_, t_, res_noaux['AUC'])
except: pass
del m3; torch.cuda.empty_cache()

print('  [4/4] Full HierAttn-Block (from nb3) ...')
res_full = results['HierAttnBlock']
print(f'        {res_full}')

# ── Print ablation table ──────────────────────────────────────────────────────
ablation_rows = [
    ('Sequence Only',         res_seq),
    ('Structural Only',       res_struct),
    ('No Auxiliary Head',     res_noaux),
    ('HierAttn-Block (Full)', res_full),
    ('DeepLog',               baseline_results['DeepLog']),
    ('LogBERT',               baseline_results['LogBERT']),
]
print()
print(f"  | {'Model Variant':<25} | {'Precision':>9} | {'Recall':>6} | {'F1':>6} | {'AUC':>6} |")
print(f"  |{'-'*27}|{'-'*11}|{'-'*8}|{'-'*8}|{'-'*8}|")
for name, m in ablation_rows:
    print(f"  | {name:<25} | {m['Precision']:>9} | {m['Recall']:>6} | {m['F1']:>6} | {m['AUC']:>6} |")


# =============================================================================
# STEP 12 — THESIS FIGURES (300 DPI)
# =============================================================================
print('\nSTEP 12 — THESIS FIGURES')
print('-'*40)

# ── Figure 1: Training curves ─────────────────────────────────────────────────
epochs_x = list(range(1, len(history['train_loss']) + 1))
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))
fig1.suptitle('HierAttn-Block — Training Dynamics', fontsize=14, fontweight='bold')

ax = axes1[0]
ax.plot(epochs_x, history['train_loss'], color=PALETTE[0], lw=2, label='Train')
ax.plot(epochs_x, history['val_loss'],   color=PALETTE[1], lw=2, ls='--', label='Val')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Loss Curves', fontweight='bold'); ax.legend(frameon=False)

ax = axes1[1]
ax.plot(epochs_x, history['train_f1'], color=PALETTE[0], lw=2, label='Train')
ax.plot(epochs_x, history['val_f1'],   color=PALETTE[1], lw=2, ls='--', label='Val')
best_ep = int(np.argmax(history['val_f1'])) + 1
ax.axvline(best_ep, color='grey', ls=':', lw=1, label=f'Best epoch ({best_ep})')
ax.set_xlabel('Epoch'); ax.set_ylabel('F1 Score')
ax.set_title('F1 Score Curves', fontweight='bold'); ax.legend(frameon=False)

plt.tight_layout()
fig1.savefig(os.path.join(FIGURE_DIR, 'fig1_training_curves.png'), dpi=300, bbox_inches='tight')
plt.close(fig1)
print('  ✅ fig1_training_curves.png')

# ── Figure 2: ROC curves ──────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 6))
ax2.plot([0,1], [0,1], 'k--', lw=1, alpha=0.5, label='Random (AUC=0.50)')
roc_order = ['DeepLog', 'LogBERT', 'Seq Only', 'Struct Only', 'No Aux Head', 'HierAttn-Block']
for i, name in enumerate(roc_order):
    if name not in all_roc: continue
    fpr_, tpr_, auc_ = all_roc[name]
    ax2.plot(fpr_, tpr_, color=PALETTE[i % len(PALETTE)], lw=2,
             label=f'{name}  (AUC={auc_:.4f})')
ax2.set_xlabel('False Positive Rate', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontsize=11)
ax2.set_title('ROC Curves — All Models (HDFS Test Set)', fontsize=13, fontweight='bold')
ax2.legend(frameon=False, fontsize=9, loc='lower right')
ax2.set_xlim([-0.02, 1.02]); ax2.set_ylim([-0.02, 1.05])
plt.tight_layout()
fig2.savefig(os.path.join(FIGURE_DIR, 'fig2_roc_curves.png'), dpi=300, bbox_inches='tight')
plt.close(fig2)
print('  ✅ fig2_roc_curves.png')

# ── Figure 3: Confusion matrix HierAttn-Block ─────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(5, 4))
cm_hier = confusion_matrix(y_test, hier_final_preds)
sns.heatmap(cm_hier, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 14, 'weight': 'bold'}, ax=ax3)
ax3.set_title('HierAttn-Block — Confusion Matrix\n(Test Set)', fontsize=13, fontweight='bold')
ax3.set_xlabel('Predicted Label', fontsize=11); ax3.set_ylabel('True Label', fontsize=11)
ax3.text(1.05, 0.5,
    f'P  = {hier_prec:.4f}\nR  = {hier_rec:.4f}\nF1 = {hier_f1:.4f}\nAUC= {hier_auc:.4f}',
    transform=ax3.transAxes, fontsize=10, va='center',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='grey'))
plt.tight_layout()
fig3.savefig(os.path.join(FIGURE_DIR, 'fig3_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close(fig3)
print('  ✅ fig3_confusion_matrix.png')

# ── Confusion matrices baselines ─────────────────────────────────────────────
for bname, bpreds, cmap_ in [('DeepLog', dl_preds_final, 'Blues'),
                               ('LogBERT', lb_preds_final, 'Oranges')]:
    fig_, ax_ = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, bpreds), annot=True, fmt='d', cmap=cmap_,
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], ax=ax_)
    ax_.set_title(f'{bname} — Confusion Matrix', fontweight='bold')
    ax_.set_xlabel('Predicted'); ax_.set_ylabel('True')
    plt.tight_layout()
    fig_.savefig(os.path.join(FIGURE_DIR, f'cm_{bname.lower()}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_)
print('  ✅ cm_deeplog.png + cm_logbert.png')

# ── Figure 4: Event saliency heatmap ─────────────────────────────────────────
MODEL_PATH = find_pkl('hierattn_best.pt')
print(f'  hierattn_best.pt   : {MODEL_PATH}')
model_viz = HierAttnBlock(VOCAB_SIZE).to(DEVICE)
model_viz.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model_viz.eval()

normal_idxs  = [i for i in range(len(feat_test)) if feat_test[i]['label'] == 0]
anomaly_idxs = [i for i in range(len(feat_test)) if feat_test[i]['label'] == 1]
selected = []
if normal_idxs:            selected.append(('Normal',  normal_idxs[0]))
if len(anomaly_idxs) >= 2: selected += [('Anomaly', anomaly_idxs[0]), ('Anomaly', anomaly_idxs[1])]
elif anomaly_idxs:         selected.append(('Anomaly', anomaly_idxs[0]))

def get_hidden(feat):
    ev = torch.tensor(feat['event_ids'],      dtype=torch.long).unsqueeze(0).to(DEVICE)
    pf = torch.tensor(feat['param_feats'],    dtype=torch.float32).unsqueeze(0).to(DEVICE)
    st = torch.tensor(feat['sin_time'],       dtype=torch.float32).unsqueeze(0).to(DEVICE)
    sf = torch.tensor(feat['struct_feats'],   dtype=torch.float32).unsqueeze(0).to(DEVICE)
    am = torch.tensor(feat['attention_mask'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, _, H = model_viz(ev, pf, st, sf, am, return_aux=False)
    return H.squeeze(0).cpu().numpy()

fig4, axes4 = plt.subplots(len(selected), 1, figsize=(12, 3 * len(selected)))
if len(selected) == 1: axes4 = [axes4]
for k, (stype, sidx) in enumerate(selected):
    feat_sel = feat_test[sidx]
    H_sel    = get_hidden(feat_sel)
    n_real   = int(feat_sel['attention_mask'].sum())
    scores   = np.linalg.norm(H_sel, axis=-1)[:n_real]
    scores   = scores / (scores.max() + 1e-8)
    im = axes4[k].imshow(scores.reshape(1, -1), aspect='auto', cmap='hot', vmin=0, vmax=1)
    axes4[k].set_xticks(range(n_real))
    axes4[k].set_xticklabels(
        [str(feat_sel['event_ids'][i]) for i in range(n_real)],
        rotation=45, ha='right', fontsize=8)
    axes4[k].set_yticks([])
    axes4[k].set_title(
        f"Session {k+1} ({stype}) — {n_real} events | "
        f"Label: {'Anomaly' if feat_sel['label']==1 else 'Normal'}",
        fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=axes4[k], fraction=0.01, pad=0.01)
fig4.suptitle(
    'HierAttn-Block — Event Saliency (Hidden State L2 Norm)\n'
    'Proxy: L2 norm of Transformer encoder output per event position',
    fontsize=12, fontweight='bold')
plt.tight_layout()
fig4.savefig(os.path.join(FIGURE_DIR, 'fig4_attention_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close(fig4)
del model_viz; torch.cuda.empty_cache()
print('  ✅ fig4_attention_heatmap.png')

# ── Figure 5: Structural feature importance ───────────────────────────────────
feat_names = [f'Template_{i+1}_count' for i in range(5)] + \
             ['size_std', 'n_unique_ips', 'session_duration',
              'max_gap', 'missing_allocate', 'repl_neq3']
print('  Computing permutation importance ...')
rf_pi = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
rf_pi.fit(X_train_struct, y_train)
pi   = permutation_importance(rf_pi, X_test_struct, y_test,
                               n_repeats=10, random_state=SEED, scoring='f1')
sidx_pi = np.argsort(pi.importances_mean)[::-1]

fig5, ax5 = plt.subplots(figsize=(9, 5))
ax5.barh([feat_names[i] for i in sidx_pi[::-1]],
         pi.importances_mean[sidx_pi[::-1]],
         xerr=pi.importances_std[sidx_pi[::-1]],
         color=PALETTE[0], ecolor='#999999', capsize=4, height=0.6)
ax5.set_xlabel('Mean Decrease in F1 (Permutation Importance)', fontsize=11)
ax5.set_title('Structural Feature Importance — Permutation Method',
              fontsize=13, fontweight='bold')
ax5.axvline(0, color='black', lw=0.8)
plt.tight_layout()
fig5.savefig(os.path.join(FIGURE_DIR, 'fig5_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close(fig5)
print('  ✅ fig5_feature_importance.png')

# ── Figure 6: Session length distribution ────────────────────────────────────
len_normal  = [int(feat_test[i]['attention_mask'].sum())
               for i in range(len(feat_test)) if feat_test[i]['label'] == 0]
len_anomaly = [int(feat_test[i]['attention_mask'].sum())
               for i in range(len(feat_test)) if feat_test[i]['label'] == 1]
fig6, ax6 = plt.subplots(figsize=(8, 5))
ax6.hist(len_normal,  bins=range(1, MAX_LEN + 2), alpha=0.6, color=PALETTE[0],
         label=f'Normal (n={len(len_normal):,})',  density=True)
ax6.hist(len_anomaly, bins=range(1, MAX_LEN + 2), alpha=0.6, color=PALETTE[1],
         label=f'Anomaly (n={len(len_anomaly):,})', density=True)
ax6.set_xlabel('Session Length (# events)', fontsize=11)
ax6.set_ylabel('Density', fontsize=11)
ax6.set_title('Session Length Distribution — Normal vs. Anomaly',
              fontsize=13, fontweight='bold')
ax6.legend(frameon=False, fontsize=10)
plt.tight_layout()
fig6.savefig(os.path.join(FIGURE_DIR, 'fig6_session_length_distribution.png'),
             dpi=300, bbox_inches='tight')
plt.close(fig6)
print('  ✅ fig6_session_length_distribution.png')

# ── Figure 7: Ablation bar chart ──────────────────────────────────────────────
abl_names  = ['DeepLog', 'LogBERT', 'Seq Only', 'Struct Only', 'No Aux Head', 'HierAttn-Block (Full)']
abl_f1s    = [
    baseline_results['DeepLog']['F1'],
    baseline_results['LogBERT']['F1'],
    res_seq['F1'],
    res_struct['F1'],
    res_noaux['F1'],
    res_full['F1'],
]
abl_colors = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3], PALETTE[4],
              PALETTE[1] if res_full['F1'] < baseline_results['LogBERT']['F1'] else '#27AE60']

fig7, ax7 = plt.subplots(figsize=(10, 5))
bars = ax7.bar(abl_names, abl_f1s, color=abl_colors, width=0.55, edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, abl_f1s):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax7.set_ylim(0, 1.08)
ax7.set_ylabel('F1 Score (Test Set)', fontsize=11)
ax7.set_title('Ablation Study — F1 Score Comparison', fontsize=13, fontweight='bold')
ax7.tick_params(axis='x', labelsize=9)
plt.tight_layout()
fig7.savefig(os.path.join(FIGURE_DIR, 'fig7_ablation_bar.png'), dpi=300, bbox_inches='tight')
plt.close(fig7)
print('  ✅ fig7_ablation_bar.png')


# =============================================================================
# STEP 13 — FINAL SUMMARY
# =============================================================================
hier  = results['HierAttnBlock']
deepl = baseline_results['DeepLog']
logb  = baseline_results['LogBERT']

print('\n' + '='*65)
print('  FINAL RESULTS')
print('='*65)
print(f"\n  {'Model':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
print(f"  {'-'*54}")
print(f"  {'HierAttn-Block':<22} {hier['Precision']:>10} {hier['Recall']:>8} {hier['F1']:>8} {hier['AUC']:>8}")
print(f"  {'DeepLog':<22}  {deepl['Precision']:>9} {deepl['Recall']:>8} {deepl['F1']:>8} {deepl['AUC']:>8}")
print(f"  {'LogBERT':<22}  {logb['Precision']:>9}  {logb['Recall']:>7} {logb['F1']:>8} {logb['AUC']:>8}")
print(f"\n  vs DeepLog  : F1 {'+' if hier['F1']>=deepl['F1'] else ''}{(hier['F1']-deepl['F1'])*100:.2f}%  |  "
      f"AUC {'+' if hier['AUC']>=deepl['AUC'] else ''}{(hier['AUC']-deepl['AUC'])*100:.2f}%")
print(f"  vs LogBERT  : F1 {'+' if hier['F1']>=logb['F1'] else ''}{(hier['F1']-logb['F1'])*100:.2f}%  |  "
      f"AUC {'+' if hier['AUC']>=logb['AUC'] else ''}{(hier['AUC']-logb['AUC'])*100:.2f}%")
print('='*65)

# ── Save final JSON ───────────────────────────────────────────────────────────
results_all = {
    'HierAttnBlock': hier,
    'DeepLog':       deepl,
    'LogBERT':       logb,
    'Ablation': {
        'Sequence Only':         res_seq,
        'Structural Only':       res_struct,
        'No Auxiliary Head':     res_noaux,
        'HierAttn-Block (Full)': res_full,
    }
}
results_path = os.path.join(OUTPUT_DIR, 'final_results.json')
with open(results_path, 'w') as fp:
    json.dump(results_all, fp, indent=2)

# ── File checklist ────────────────────────────────────────────────────────────
print('\nFIGURES:')
all_figs = [
    'fig1_training_curves.png',
    'fig2_roc_curves.png',
    'fig3_confusion_matrix.png',
    'fig4_attention_heatmap.png',
    'fig5_feature_importance.png',
    'fig6_session_length_distribution.png',
    'fig7_ablation_bar.png',
    'cm_deeplog.png',
    'cm_logbert.png',
]
for fname in all_figs:
    fp = os.path.join(FIGURE_DIR, fname)
    print(f'  {"✅" if os.path.exists(fp) else "❌"}  {fname}')

print(f'\n  ✅ final_results.json saved → {results_path}')
print('\n' + '='*65)
print('  PROJECT COMPLETE')
print('='*65)
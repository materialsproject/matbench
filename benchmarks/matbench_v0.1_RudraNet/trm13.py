"""
╔══════════════════════════════════════════════════════════════════════╗
║  TRM-MatSci V13 — 2-Layer SA + Multi-Seed Ensemble                   ║
║  Dataset: matbench_steels │ 312 samples │ 5-Fold CV + inner holdout          ║
║                                                                      ║
║  V13A  2-Layer Self-Attention + Standard Deep Supervision            ║
║        d_attn=64, nhead=4, d_hidden=96, ff_dim=150, 20 steps       ║
║        Expanded features (Magpie + Mat2Vec + Extra descriptors)      ║
║        2nd SA layer for higher-order property interactions           ║
║        5-seed ensemble (avg predictions across seeds)                ║
║                                                                      ║
║  V13B  Same 2-Layer SA + Confidence-Weighted Deep Supervision        ║
║        22 steps, confidence_head learns which step to trust          ║
║        5-seed ensemble (avg predictions across seeds)                ║
║                                                                      ║
║  All models: Deep Supervision + SWA + AdamW + 300 epochs            ║
║  Baseline: V12A = 95.99 MPa (current best)                          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, copy, json, time, logging, warnings, shutil, urllib.request, hashlib
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from gensim.models import Word2Vec

logging.basicConfig(level=logging.INFO, format='%(name)s │ %(message)s')
log = logging.getLogger("TRM13")

# Seeds for multi-seed ensemble
SEEDS = [42, 123, 7, 0, 99]
N_SEEDS = len(SEEDS)

BASELINES = {
    'TPOT-Mat':       79.9468,
    'AutoML-Mat':     82.3043,
    'MODNet':         87.7627,
    'RF-SCM/Magpie': 103.5125,
    'V12A (best)':    95.9900,
    'V11B':          102.3003,
    'V10A':          103.2867,
    'CrabNet':       107.3160,
    'Darwin':        123.2932,
}


# ══════════════════════════════════════════════════════════════════════
# 1. FEATURIZER + DATASET
# ══════════════════════════════════════════════════════════════════════

class ExpandedFeaturizer:
    """Magpie (22 props × 6 stats) + Extra matminer descriptors + Mat2Vec (200d).

    Extra descriptors: ElementFraction, Stoichiometry, ValenceOrbital,
    IonProperty, BandCenter — all concatenated as a flat vector between
    the Magpie block and Mat2Vec.
    """
    GCS = "https://storage.googleapis.com/mat2vec/"
    FILES = ["pretrained_embeddings",
             "pretrained_embeddings.wv.vectors.npy",
             "pretrained_embeddings.trainables.syn1neg.npy"]

    def __init__(self, cache="mat2vec_cache"):
        from matminer.featurizers.composition import (
            ElementFraction, Stoichiometry, ValenceOrbital,
            IonProperty, BandCenter
        )
        from matminer.featurizers.base import MultipleFeaturizer

        self.ep_magpie = ElementProperty.from_preset("magpie")
        self.n_mg = len(self.ep_magpie.feature_labels())

        self.extra_feats = MultipleFeaturizer([
            ElementFraction(),
            Stoichiometry(),
            ValenceOrbital(),
            IonProperty(),
            BandCenter(),
        ])
        self.n_extra = None   # detected at featurize time

        self.scaler = None
        os.makedirs(cache, exist_ok=True)

        # Bind binary Mat2Vec inputs to the SHA-256 manifest captured from
        # the official upstream Google Cloud Storage objects (October 2026).
        # The historical Kaggle cache could not be independently recovered.
        manifest_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mat2vec_sha256.json"
        )
        with open(manifest_file, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("source") != self.GCS or set(manifest.get("files", {})) != set(self.FILES):
            raise RuntimeError("Invalid or incomplete Mat2Vec integrity manifest")

        def _mat2vec_sha256(path):
            h = hashlib.sha256()
            with open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        for f in self.FILES:
            expected = manifest["files"][f]["sha256"]
            p = os.path.join(cache, f)
            if not os.path.isfile(p):
                log.info(f"  Downloading verified upstream Mat2Vec asset {f}...")
                temporary = p + ".download"
                try:
                    urllib.request.urlretrieve(self.GCS + f, temporary)
                    digest = _mat2vec_sha256(temporary)
                    if digest != expected:
                        raise RuntimeError(f"Mat2Vec SHA-256 mismatch for download {f}")
                    os.replace(temporary, p)
                finally:
                    if os.path.exists(temporary):
                        os.remove(temporary)
            # Never load pre-existing cache files without verifying them.
            digest = _mat2vec_sha256(p)
            if digest != expected:
                raise RuntimeError(
                    f"Mat2Vec cache checksum mismatch: {f}. "
                    "Remove the mismatched cached file to redownload."
                )
            if os.path.getsize(p) != manifest["files"][f]["size_bytes"]:
                raise RuntimeError(f"Mat2Vec file size mismatch: {f}")

        self.m2v = Word2Vec.load(os.path.join(cache, "pretrained_embeddings"))
        self.emb = {w: self.m2v.wv[w] for w in self.m2v.wv.index_to_key}

    def _pool(self, c):
        v, t = np.zeros(200, np.float32), 0.0
        for s, f in c.get_el_amt_dict().items():
            if s in self.emb: v += f * self.emb[s]; t += f
        return v / max(t, 1e-8)

    def featurize_all(self, comps):
        out = []
        for c in tqdm(comps, desc="  Featurizing (expanded)", leave=False):
            try: mg = np.array(self.ep_magpie.featurize(c), np.float32)
            except: mg = np.zeros(self.n_mg, np.float32)

            try:
                ex = np.array(self.extra_feats.featurize(c), np.float32)
            except:
                ex = np.zeros(self.n_extra or 200, np.float32)

            if self.n_extra is None:
                self.n_extra = len(ex)
                log.info(f"Expanded features: {self.n_mg} Magpie + "
                         f"{self.n_extra} Extra + 200 Mat2Vec = "
                         f"{self.n_mg + self.n_extra + 200}d")

            out.append(np.concatenate([
                np.nan_to_num(mg, nan=0.0),
                np.nan_to_num(ex, nan=0.0),
                self._pool(c)
            ]))
        return np.array(out)

    def fit_scaler(self, X): self.scaler = StandardScaler().fit(X)
    def transform(self, X):
        if not self.scaler: return X
        return np.nan_to_num(self.scaler.transform(X), nan=0.0).astype(np.float32)


class DSData(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(np.array(y, np.float32))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ══════════════════════════════════════════════════════════════════════
# 2. MODELS — with 2-Layer Self-Attention
# ══════════════════════════════════════════════════════════════════════

class DeepHybridTRM(nn.Module):
    """V13A: 2-Layer SA Hybrid-TRM with Standard Deep Supervision.

    Key difference from V12A's HybridTRM:
      - TWO self-attention layers (SA1 → FF1 → SA2 → FF2 → CA)
      - Each SA layer has its own residual + LayerNorm + FF block
      - This enables higher-order property interaction modeling
        (e.g., "correlation between electronegativity-range AND
         atomic-radius-mean" requires composing two rounds of attention)
      - Cross-attention (CA) to Mat2Vec context remains after SA stack

    Everything else (MLP reasoning loop, deep supervision, SWA)
    is identical to V12A.
    """
    def __init__(self, n_props=22, stat_dim=6, n_extra=0, mat2vec_dim=200,
                 d_attn=64, nhead=4, d_hidden=96, ff_dim=150,
                 dropout=0.2, max_steps=20, **kw):
        super().__init__()
        self.max_steps, self.D = max_steps, d_hidden
        self.n_props, self.stat_dim = n_props, stat_dim
        self.n_extra = n_extra

        # ── Attention feature extractor (2-Layer SA) ──────────────────
        self.tok_proj = nn.Sequential(
            nn.Linear(stat_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())
        self.m2v_proj = nn.Sequential(
            nn.Linear(mat2vec_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())

        # Self-Attention Layer 1
        self.sa1 = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa1_n = nn.LayerNorm(d_attn)
        self.sa1_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_attn*2, d_attn))
        self.sa1_fn = nn.LayerNorm(d_attn)

        # Self-Attention Layer 2 (NEW — captures higher-order interactions)
        self.sa2 = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa2_n = nn.LayerNorm(d_attn)
        self.sa2_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_attn*2, d_attn))
        self.sa2_fn = nn.LayerNorm(d_attn)

        # Cross-Attention to Mat2Vec context (after SA stack)
        self.ca = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.ca_n = nn.LayerNorm(d_attn)

        # Pool with optional extra feature injection
        pool_in = d_attn + (n_extra if n_extra > 0 else 0)
        self.pool = nn.Sequential(
            nn.Linear(pool_in, d_hidden), nn.LayerNorm(d_hidden), nn.GELU())

        # MLP-TRM recursive reasoning (shared weights)
        self.z_up = nn.Sequential(
            nn.Linear(d_hidden*3, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_hidden), nn.LayerNorm(d_hidden))
        self.y_up = nn.Sequential(
            nn.Linear(d_hidden*2, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_hidden), nn.LayerNorm(d_hidden))
        self.head = nn.Linear(d_hidden, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _attention(self, x):
        B = x.size(0)
        mg_dim = self.n_props * self.stat_dim
        mg = x[:, :mg_dim]

        if self.n_extra > 0:
            extra = x[:, mg_dim:mg_dim + self.n_extra]
            m2v = x[:, mg_dim + self.n_extra:]
        else:
            extra = None
            m2v = x[:, mg_dim:]

        tok = self.tok_proj(mg.view(B, self.n_props, self.stat_dim))
        ctx = self.m2v_proj(m2v).unsqueeze(1)

        # SA Layer 1: learn pairwise property interactions
        tok = self.sa1_n(tok + self.sa1(tok, tok, tok)[0])
        tok = self.sa1_fn(tok + self.sa1_ff(tok))

        # SA Layer 2: learn higher-order property interactions
        tok = self.sa2_n(tok + self.sa2(tok, tok, tok)[0])
        tok = self.sa2_fn(tok + self.sa2_ff(tok))

        # Cross-Attention to Mat2Vec chemistry context
        tok = self.ca_n(tok + self.ca(tok, ctx, ctx)[0])

        pooled = tok.mean(dim=1)  # [B, d_attn]

        if extra is not None:
            pooled = torch.cat([pooled, extra], dim=-1)

        return self.pool(pooled)  # [B, d_hidden]

    def forward(self, x, deep_supervision=False, return_trajectory=False):
        B = x.size(0)
        xp = self._attention(x)
        z = torch.zeros(B, self.D, device=x.device)
        y = torch.zeros(B, self.D, device=x.device)
        step_preds = []
        for _ in range(self.max_steps):
            z = z + self.z_up(torch.cat([xp, y, z], -1))
            y = y + self.y_up(torch.cat([y, z], -1))
            step_preds.append(self.head(y).squeeze(1))
        if deep_supervision:
            return step_preds
        elif return_trajectory:
            return step_preds[-1], step_preds
        else:
            return step_preds[-1]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepConfidenceHybridTRM(nn.Module):
    """V13B: 2-Layer SA Hybrid-TRM with Confidence-Weighted Deep Supervision.

    Same 2-layer SA feature extractor as DeepHybridTRM, but with:
      - confidence_head that learns which recursion step to trust
      - Final prediction = softmax(confidence) · step_preds
      - No ponder cost (avoids V11C's failure)
      - 22 recursion steps (vs 20 for V13A)
    """
    def __init__(self, n_props=22, stat_dim=6, n_extra=0, mat2vec_dim=200,
                 d_attn=64, nhead=4, d_hidden=96, ff_dim=150,
                 dropout=0.2, max_steps=22, **kw):
        super().__init__()
        self.max_steps, self.D = max_steps, d_hidden
        self.n_props, self.stat_dim = n_props, stat_dim
        self.n_extra = n_extra

        # ── Attention feature extractor (2-Layer SA) ──────────────────
        self.tok_proj = nn.Sequential(
            nn.Linear(stat_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())
        self.m2v_proj = nn.Sequential(
            nn.Linear(mat2vec_dim, d_attn), nn.LayerNorm(d_attn), nn.GELU())

        # Self-Attention Layer 1
        self.sa1 = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa1_n = nn.LayerNorm(d_attn)
        self.sa1_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_attn*2, d_attn))
        self.sa1_fn = nn.LayerNorm(d_attn)

        # Self-Attention Layer 2 (higher-order interactions)
        self.sa2 = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.sa2_n = nn.LayerNorm(d_attn)
        self.sa2_ff = nn.Sequential(
            nn.Linear(d_attn, d_attn*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_attn*2, d_attn))
        self.sa2_fn = nn.LayerNorm(d_attn)

        # Cross-Attention to Mat2Vec context
        self.ca = nn.MultiheadAttention(
            d_attn, nhead, dropout=dropout, batch_first=True)
        self.ca_n = nn.LayerNorm(d_attn)

        # Pool with optional extra feature injection
        pool_in = d_attn + (n_extra if n_extra > 0 else 0)
        self.pool = nn.Sequential(
            nn.Linear(pool_in, d_hidden), nn.LayerNorm(d_hidden), nn.GELU())

        # MLP-TRM recursive reasoning (shared weights)
        self.z_up = nn.Sequential(
            nn.Linear(d_hidden*3, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_hidden), nn.LayerNorm(d_hidden))
        self.y_up = nn.Sequential(
            nn.Linear(d_hidden*2, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_hidden), nn.LayerNorm(d_hidden))
        self.head = nn.Linear(d_hidden, 1)

        # ── Confidence head: learns which step to trust ──────────────
        self.confidence_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(),
            nn.Linear(d_hidden // 2, 1))  # raw logit, softmaxed later

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        with torch.no_grad():
            nn.init.zeros_(self.confidence_head[-1].bias)

    def _attention(self, x):
        B = x.size(0)
        mg_dim = self.n_props * self.stat_dim
        mg = x[:, :mg_dim]

        if self.n_extra > 0:
            extra = x[:, mg_dim:mg_dim + self.n_extra]
            m2v = x[:, mg_dim + self.n_extra:]
        else:
            extra = None
            m2v = x[:, mg_dim:]

        tok = self.tok_proj(mg.view(B, self.n_props, self.stat_dim))
        ctx = self.m2v_proj(m2v).unsqueeze(1)

        # SA Layer 1
        tok = self.sa1_n(tok + self.sa1(tok, tok, tok)[0])
        tok = self.sa1_fn(tok + self.sa1_ff(tok))

        # SA Layer 2
        tok = self.sa2_n(tok + self.sa2(tok, tok, tok)[0])
        tok = self.sa2_fn(tok + self.sa2_ff(tok))

        # Cross-Attention
        tok = self.ca_n(tok + self.ca(tok, ctx, ctx)[0])

        pooled = tok.mean(dim=1)

        if extra is not None:
            pooled = torch.cat([pooled, extra], dim=-1)

        return self.pool(pooled)

    def forward(self, x, deep_supervision=False, return_confidence=False):
        """Forward pass.

        Returns:
            deep_supervision=True:  (step_preds, confidence_logits)
            deep_supervision=False, return_confidence=False:
                weighted_pred: [B] confidence-weighted prediction
            deep_supervision=False, return_confidence=True:
                (weighted_pred, confidence_weights): [B], [B, max_steps]
        """
        B = x.size(0)
        xp = self._attention(x)
        z = torch.zeros(B, self.D, device=x.device)
        y = torch.zeros(B, self.D, device=x.device)

        step_preds = []
        conf_logits = []

        for _ in range(self.max_steps):
            z = z + self.z_up(torch.cat([xp, y, z], -1))
            y = y + self.y_up(torch.cat([y, z], -1))
            step_preds.append(self.head(y).squeeze(1))
            conf_logits.append(self.confidence_head(y).squeeze(1))

        conf_logits = torch.stack(conf_logits, dim=1)  # [B, max_steps]

        if deep_supervision:
            return step_preds, conf_logits

        # Confidence-weighted prediction
        conf_weights = F.softmax(conf_logits, dim=1)  # [B, max_steps]
        preds_stack = torch.stack(step_preds, dim=1)   # [B, max_steps]
        weighted_pred = (preds_stack * conf_weights).sum(dim=1)  # [B]

        if return_confidence:
            return weighted_pred, conf_weights
        return weighted_pred

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════
# 3. LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def deep_supervision_loss(step_preds, targets):
    """Linear-weighted L1 loss across all recursion steps."""
    n = len(step_preds)
    weights = [(i + 1) for i in range(n)]
    total_w = sum(weights)
    loss = 0.0
    for pred, w in zip(step_preds, weights):
        loss += (w / total_w) * F.l1_loss(pred, targets)
    return loss


def confidence_ds_loss(step_preds, targets, conf_logits):
    """Advanced Deep Supervision: standard DS + confidence-weighted L1.

    Components:
    1. Standard linear-weighted deep supervision on all steps
    2. L1 loss on the confidence-weighted final prediction
    """
    ds = deep_supervision_loss(step_preds, targets)

    conf_weights = F.softmax(conf_logits, dim=1)  # [B, max_steps]
    preds_stack = torch.stack(step_preds, dim=1)   # [B, max_steps]
    weighted_pred = (preds_stack * conf_weights).sum(dim=1)
    conf_loss = F.l1_loss(weighted_pred, targets)

    return ds + conf_loss


# ══════════════════════════════════════════════════════════════════════
# 4. UTILS + TRAINING
# ══════════════════════════════════════════════════════════════════════

def strat_split(targets, val_size=0.15, seed=42):
    bins = np.percentile(targets, [25, 50, 75])
    lbl = np.digitize(targets, bins)
    tr, vl = [], []
    rng = np.random.RandomState(seed)
    for b in range(4):
        m = np.where(lbl == b)[0]
        if len(m) == 0: continue
        n = max(1, int(len(m) * val_size))
        c = rng.choice(m, n, replace=False)
        vl.extend(c.tolist()); tr.extend(np.setdiff1d(m, c).tolist())
    return np.array(tr), np.array(vl)


def train_fold_standard(model, tr_dl, vl_dl, device,
                        epochs=300, swa_start=200, fold=1, name=""):
    """Training with standard deep supervision."""
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=swa_start, eta_min=1e-4)
    swa_m = AveragedModel(model)
    swa_s = SWALR(opt, swa_lr=5e-4)
    swa_on = False
    best_v, best_w = float('inf'), copy.deepcopy(model.state_dict())
    hist = {'train': [], 'val': []}

    pbar = tqdm(range(epochs), desc=f"  [{name}] F{fold}/5",
                leave=False, ncols=120)
    for ep in pbar:
        model.train(); tl = 0.0
        for bx, by in tr_dl:
            bx, by = bx.to(device), by.to(device)
            step_preds = model(bx, deep_supervision=True)
            loss = deep_supervision_loss(step_preds, by)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += F.l1_loss(step_preds[-1], by).item() * len(by)
        tl /= len(tr_dl.dataset)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for bx, by in vl_dl:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                vl += F.l1_loss(pred, by).item() * len(by)
        vl /= len(vl_dl.dataset)
        hist['train'].append(tl); hist['val'].append(vl)

        if ep < swa_start:
            sch.step()
            if vl < best_v: best_v, best_w = vl, copy.deepcopy(model.state_dict())
        else:
            if not swa_on: swa_on = True
            swa_m.update_parameters(model); swa_s.step()

        pbar.set_postfix(Tr=f'{tl:.1f}', Val=f'{vl:.1f}',
                        Best=f'{best_v:.1f}', Ph='SWA' if swa_on else 'COS')

    if swa_on:
        update_bn(tr_dl, swa_m, device=device)
        model.load_state_dict(swa_m.module.state_dict())
    else:
        model.load_state_dict(best_w)
    return best_v, model, hist


def train_fold_confidence(model, tr_dl, vl_dl, device,
                          epochs=300, swa_start=200, fold=1, name=""):
    """Training with confidence-weighted deep supervision."""
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=swa_start, eta_min=1e-4)
    swa_m = AveragedModel(model)
    swa_s = SWALR(opt, swa_lr=5e-4)
    swa_on = False
    best_v, best_w = float('inf'), copy.deepcopy(model.state_dict())
    hist = {'train': [], 'val': []}

    pbar = tqdm(range(epochs), desc=f"  [{name}] F{fold}/5",
                leave=False, ncols=120)
    for ep in pbar:
        model.train(); tl = 0.0
        for bx, by in tr_dl:
            bx, by = bx.to(device), by.to(device)
            step_preds, conf_logits = model(bx, deep_supervision=True)
            loss = confidence_ds_loss(step_preds, by, conf_logits)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # Track confidence-weighted MAE for display
            with torch.no_grad():
                cw = F.softmax(conf_logits, dim=1)
                ps = torch.stack(step_preds, dim=1)
                wp = (ps * cw).sum(dim=1)
                tl += F.l1_loss(wp, by).item() * len(by)
        tl /= len(tr_dl.dataset)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for bx, by in vl_dl:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)  # uses confidence-weighted by default
                vl += F.l1_loss(pred, by).item() * len(by)
        vl /= len(vl_dl.dataset)
        hist['train'].append(tl); hist['val'].append(vl)

        if ep < swa_start:
            sch.step()
            if vl < best_v: best_v, best_w = vl, copy.deepcopy(model.state_dict())
        else:
            if not swa_on: swa_on = True
            swa_m.update_parameters(model); swa_s.step()

        pbar.set_postfix(Tr=f'{tl:.1f}', Val=f'{vl:.1f}',
                        Best=f'{best_v:.1f}', Ph='SWA' if swa_on else 'COS')

    if swa_on:
        update_bn(tr_dl, swa_m, device=device)
        model.load_state_dict(swa_m.module.state_dict())
    else:
        model.load_state_dict(best_w)
    return best_v, model, hist


def predict(model, dl, device):
    model.eval(); preds = []
    with torch.no_grad():
        for bx, _ in dl:
            preds.append(model(bx.to(device)).cpu())
    return torch.cat(preds)


def predict_confidence(model, dl, device):
    """Predict using confidence model, also return per-step weights."""
    model.eval()
    all_preds, all_weights = [], []
    with torch.no_grad():
        for bx, _ in dl:
            pred, weights = model(bx.to(device), return_confidence=True)
            all_preds.append(pred.cpu())
            all_weights.append(weights.cpu())
    return torch.cat(all_preds), torch.cat(all_weights)


def get_targets(dl):
    tgts = []
    for _, by in dl: tgts.append(by)
    return torch.cat(tgts)


# ══════════════════════════════════════════════════════════════════════
# 5. MAIN BENCHMARK — Multi-Seed Ensemble
# ══════════════════════════════════════════════════════════════════════

def run_benchmark():
    t0 = time.time()
    print("\n" + "═"*72)
    print("  TRM-MatSci V13 │ 2-Layer SA + Multi-Seed Ensemble │ matbench_steels")
    print("  V13A: 2-Layer SA + expanded features + standard DS (5-seed ensemble)")
    print("  V13B: 2-Layer SA + expanded features + confidence DS (5-seed ensemble)")
    print(f"  Seeds: {SEEDS}")
    print("═"*72 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        log.info(f"GPU: {torch.cuda.get_device_name(0)}  "
                 f"({torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    log.info("Loading matbench_steels...")
    from matminer.datasets import load_dataset
    df = load_dataset("matbench_steels")
    comps_raw = df['composition'].tolist()
    targets_all = np.array(df['yield strength'].tolist(), np.float32)
    comps_all = [Composition(c) for c in comps_raw]

    # ── FEATURIZE ─────────────────────────────────────────────────────
    log.info("Computing EXPANDED features...")
    feat = ExpandedFeaturizer()
    X_all = feat.featurize_all(comps_all)
    n_extra = feat.n_extra
    log.info(f"Features: {X_all.shape} (n_extra={n_extra})")

    kfold = KFold(n_splits=5, shuffle=True, random_state=18012019)
    folds = list(kfold.split(comps_all))
    os.makedirs('trm_models_v13', exist_ok=True)
    dl_kw = dict(batch_size=32, num_workers=0)

    # ── CONFIGS ───────────────────────────────────────────────────────
    shared_kw = dict(n_props=22, stat_dim=6, n_extra=n_extra,
                     mat2vec_dim=200, d_attn=64, nhead=4,
                     d_hidden=96, ff_dim=150, dropout=0.2)

    configs = {
        'V13A-2xSA-StdDS': {
            'model_cls': DeepHybridTRM,
            'model_kw': {**shared_kw, 'max_steps': 20},
            'train_fn': train_fold_standard,
            'predict_fn': predict,
            'is_confidence': False,
        },
        'V13B-2xSA-ConfDS': {
            'model_cls': DeepConfidenceHybridTRM,
            'model_kw': {**shared_kw, 'max_steps': 22},
            'train_fn': train_fold_confidence,
            'predict_fn': None,  # uses predict_confidence
            'is_confidence': True,
        },
    }

    # Print param counts
    print(f"\n  {'Config':<24} {'Params':>10} {'Steps':>8}  {'Seeds':>6}")
    print(f"  {'─'*54}")
    for cname, cfg in configs.items():
        _m = cfg['model_cls'](**cfg['model_kw'])
        np_ = _m.count_parameters(); del _m
        cfg['n_params'] = np_
        steps = cfg['model_kw']['max_steps']
        print(f"  {cname:<24} {np_:>10,} {steps:>8}  {N_SEEDS:>6}")
    print()

    # ── TRAIN + EVALUATE (Multi-Seed) ─────────────────────────────────
    all_results = {}
    all_hists = {}
    all_conf_weights = {}
    # Raw V13A ensemble outputs, exported by MBID for official Matbench recording.
    v13a_export = {}

    for cname, cfg in configs.items():
        print(f"\n{'▓'*72}")
        print(f"  {cname} — {N_SEEDS}-Seed Ensemble")
        print(f"{'▓'*72}")

        # Store per-seed, per-fold predictions and MAEs
        seed_fold_preds = {s: {} for s in SEEDS}    # seed -> {fold_idx: preds_tensor}
        seed_fold_maes = {s: [] for s in SEEDS}     # seed -> [mae_f1, ..., mae_f5]
        fold_hists = []                              # collect from first seed only
        fold_conf_w = []                             # collect from first seed only

        for si, seed in enumerate(SEEDS):
            print(f"\n  ╔═══ Seed {seed} ({si+1}/{N_SEEDS}) ═══╗")

            for fi, (tv_i, te_i) in enumerate(folds):
                print(f"\n  ── [{cname} seed={seed}] Fold {fi+1}/5 {'─'*30}")

                tri, vli = strat_split(targets_all[tv_i], 0.15, seed+fi)
                feat.fit_scaler(X_all[tv_i][tri])
                tr_s = feat.transform(X_all[tv_i][tri])
                vl_s = feat.transform(X_all[tv_i][vli])
                te_s = feat.transform(X_all[te_i])

                pin = device.type == 'cuda'
                tr_dl = DataLoader(DSData(tr_s, targets_all[tv_i][tri]), shuffle=True,
                                   pin_memory=pin, **dl_kw)
                vl_dl = DataLoader(DSData(vl_s, targets_all[tv_i][vli]), shuffle=False,
                                   pin_memory=pin, **dl_kw)
                te_dl = DataLoader(DSData(te_s, targets_all[te_i]), shuffle=False,
                                   pin_memory=pin, **dl_kw)
                te_tgt = get_targets(te_dl)

                torch.manual_seed(seed + fi); np.random.seed(seed + fi)
                if device.type == 'cuda': torch.cuda.manual_seed(seed + fi)

                model = cfg['model_cls'](**cfg['model_kw']).to(device)
                bv, model, hist = cfg['train_fn'](model, tr_dl, vl_dl, device,
                                                   fold=fi+1,
                                                   name=f"{cname}[s{seed}]")

                # Save hist only for first seed
                if si == 0:
                    fold_hists.append(hist)

                # Predict
                if cfg['is_confidence']:
                    pred, conf_w = predict_confidence(model, te_dl, device)
                    if si == 0:
                        fold_conf_w.append(conf_w)
                    avg_peak = conf_w.argmax(dim=1).float().mean().item() + 1
                    mae = F.l1_loss(pred, te_tgt).item()
                    log.info(f"  [s{seed}] F{fi+1}: MAE={mae:.2f}  "
                             f"(val {bv:.2f}, avg peak step={avg_peak:.1f})")
                else:
                    pred = cfg['predict_fn'](model, te_dl, device)
                    mae = F.l1_loss(pred, te_tgt).item()
                    log.info(f"  [s{seed}] F{fi+1}: MAE={mae:.2f}  (val {bv:.2f})")

                seed_fold_preds[seed][fi] = pred
                seed_fold_maes[seed].append(mae)

                torch.save({'model_state': model.state_dict(), 'test_mae': mae,
                            'config': cname, 'seed': seed},
                           f'trm_models_v13/{cname}_seed{seed}_fold{fi+1}.pt')

                # Free GPU memory
                del model; torch.cuda.empty_cache() if device.type == 'cuda' else None

            seed_avg = float(np.mean(seed_fold_maes[seed]))
            print(f"  ╚═══ Seed {seed} avg: {seed_avg:.2f} MPa ═══╝")

        # ── Compute ensemble predictions ──────────────────────────────
        ensemble_fold_maes = []
        for fi, (tv_i, te_i) in enumerate(folds):
            te_tgt_np = targets_all[te_i]
            te_tgt_t = torch.tensor(te_tgt_np, dtype=torch.float32)

            # Average predictions across all seeds for this fold
            all_seed_preds = torch.stack([seed_fold_preds[s][fi] for s in SEEDS])
            ensemble_pred = all_seed_preds.mean(dim=0)
            if cname == 'V13A-2xSA-StdDS':
                # IDs correspond to original matminer row positions, matching
                # Matbench's mb-steels-XXX ID assignment for this dataset.
                v13a_export[f'fold_{fi}'] = {
                    f'mb-steels-{int(row_index) + 1:03d}': float(value)
                    for row_index, value in zip(te_i, ensemble_pred.tolist())
                }

            ens_mae = F.l1_loss(ensemble_pred, te_tgt_t).item()
            ensemble_fold_maes.append(ens_mae)

        ens_avg = float(np.mean(ensemble_fold_maes))
        ens_std = float(np.std(ensemble_fold_maes))

        # Also compute per-seed averages for reporting
        per_seed_avgs = {s: float(np.mean(seed_fold_maes[s])) for s in SEEDS}
        best_single_seed = min(per_seed_avgs.items(), key=lambda x: x[1])

        all_results[cname] = {
            'avg': ens_avg, 'std': ens_std, 'folds': ensemble_fold_maes,
            'params': cfg['n_params'],
            'per_seed_avgs': per_seed_avgs,
            'per_seed_folds': {str(s): seed_fold_maes[s] for s in SEEDS},
            'best_single_seed': best_single_seed[0],
            'best_single_mae': best_single_seed[1],
        }
        all_hists[cname] = fold_hists
        if fold_conf_w:
            all_conf_weights[cname] = fold_conf_w

        print(f"\n  ═══ {cname} ═══")
        print(f"      Ensemble ({N_SEEDS}-seed avg): {ens_avg:.4f} ±{ens_std:.4f} MPa")
        print(f"      Best single seed ({best_single_seed[0]}): "
              f"{best_single_seed[1]:.4f} MPa")
        for s in SEEDS:
            print(f"      Seed {s:>3}: {per_seed_avgs[s]:.2f} MPa  "
                  f"folds={[f'{m:.1f}' for m in seed_fold_maes[s]]}")

    # ══════════════════════════════════════════════════════════════════
    # FINAL RESULTS
    # ══════════════════════════════════════════════════════════════════

    tt = time.time() - t0
    print(f"\n{'═'*72}")
    print(f"  FINAL LEADERBOARD — matbench_steels V13 (5-Fold Avg MAE)")
    print(f"{'═'*72}")
    print(f"  {'Model':<26} {'Params':>10} {'MAE(MPa)':>10} {'±Std':>8}  Notes")
    print(f"  {'─'*72}")
    for n, r in sorted(all_results.items(), key=lambda x: x[1]['avg']):
        tag = ("  ← BEATS MODNet 🏆" if r['avg'] < 87.76 else
               "  ← BEATS V12A ✓"  if r['avg'] < 95.99 else
               "  ← BEATS RF-SCM ✓"  if r['avg'] < 103.51 else
               "  ← BEATS DARWIN ✓"  if r['avg'] < 123.29 else "")
        print(f"  {n+' (ens)':<26} {r['params']:>9,} "
              f"{r['avg']:>10.4f} {r['std']:>8.4f}{tag}")
        print(f"  {n+' (best 1)':<26} {'':>10} "
              f"{r['best_single_mae']:>10.4f} {'':>8}  seed={r['best_single_seed']}")
    print(f"  {'─'*72}")
    for bn, bv in sorted(BASELINES.items(), key=lambda x: x[1]):
        print(f"  {bn:<26} {'baseline':>10} {bv:>10.4f}")
    print(f"\n  Total time: {tt/60:.1f} minutes  ({N_SEEDS} seeds × 2 configs × 5 folds)")

    # Per-fold ensemble breakdown
    print(f"\n{'═'*72}")
    print(f"  PER-FOLD ENSEMBLE BREAKDOWN")
    print(f"{'═'*72}")
    cnames = list(all_results.keys())
    header = f"  {'Fold':<6}"
    for cn in cnames:
        header += f" {cn:>20}"
    print(header)
    print(f"  {'─'*52}")
    for fi in range(5):
        row = f"  {fi+1:<6}"
        for cn in cnames:
            row += f" {all_results[cn]['folds'][fi]:>20.2f}"
        print(row)

    # Per-seed breakdown
    print(f"\n{'═'*72}")
    print(f"  PER-SEED BREAKDOWN")
    print(f"{'═'*72}")
    for cn in cnames:
        r = all_results[cn]
        print(f"\n  {cn}:")
        header = f"    {'Seed':<6}"
        for fi in range(5):
            header += f" {'F'+str(fi+1):>8}"
        header += f" {'Avg':>8}"
        print(header)
        print(f"    {'─'*52}")
        for s in SEEDS:
            row = f"    {s:<6}"
            for mae in r['per_seed_folds'][str(s)]:
                row += f" {mae:>8.2f}"
            row += f" {r['per_seed_avgs'][s]:>8.2f}"
            print(row)
        print(f"    {'─'*52}")
        row = f"    {'ENS':<6}"
        for mae in r['folds']:
            row += f" {mae:>8.2f}"
        row += f" {r['avg']:>8.2f}"
        print(row)

    # Confidence stats
    if all_conf_weights:
        print(f"\n  Confidence Step Selection Summary:")
        for cn, fw_list in all_conf_weights.items():
            all_w = torch.cat(fw_list, dim=0)
            avg_w = all_w.mean(dim=0)
            peak_step = avg_w.argmax().item() + 1
            avg_peak = all_w.argmax(dim=1).float().mean().item() + 1
            print(f"    {cn}: avg peak step={avg_peak:.1f}, "
                  f"population peak=step {peak_step}")
    print()

    # Preserve raw V13A predictions so a separate environment with Matbench
    # can validate and serialize the exact sample-to-prediction mapping.
    if len(v13a_export) != 5:
        raise RuntimeError('Expected 5 V13A prediction folds for export')
    with open('trm_models_v13/v13a_fold_predictions_by_mbid.json', 'w') as f:
        json.dump(v13a_export, f, indent=2)

    generate_plots(all_results, all_hists, all_conf_weights)
    save_summary(all_results, all_hists, all_conf_weights, tt)
    return all_results


# ══════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ══════════════════════════════════════════════════════════════════════

PAL = {'V13A-2xSA-StdDS': '#1565C0', 'V13B-2xSA-ConfDS': '#E65100'}

def generate_plots(all_results, all_hists, all_conf_weights):
    names = list(all_results.keys())
    avgs = [all_results[n]['avg'] for n in names]
    stds = [all_results[n]['std'] for n in names]
    cols = [PAL.get(n, '#888') for n in names]

    fig = plt.figure(figsize=(22, 18))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # ── Plot 1: Bar chart vs baselines ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    # Show both ensemble and best-single-seed bars
    x_pos = np.arange(len(names))
    w = 0.35
    ens_bars = ax1.bar(x_pos - w/2, avgs, w, yerr=stds, capsize=6,
                       color=cols, alpha=0.88, edgecolor='white',
                       linewidth=1.5, label='Ensemble')
    best_singles = [all_results[n]['best_single_mae'] for n in names]
    single_bars = ax1.bar(x_pos + w/2, best_singles, w, capsize=6,
                          color=cols, alpha=0.45, edgecolor='white',
                          linewidth=1.5, label='Best Single Seed',
                          hatch='//')

    for bv, c, ls, lb in [
        (87.76, '#F57F17', '--', 'MODNet (87.76)'),
        (95.99, '#4CAF50', '-.', 'V12A (95.99)'),
        (102.30, '#9E9E9E', '-.', 'V11B (102.30)'),
        (103.51, '#B0BEC5', ':', 'RF-SCM (103.51)'),
        (107.32, '#FF9800', ':', 'CrabNet (107.32)'),
    ]:
        ax1.axhline(bv, color=c, linestyle=ls, linewidth=1.8, label=lb, alpha=0.85)
    for bar, m, s in zip(ens_bars, avgs, stds):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+1,
                 f'{m:.1f}', ha='center', fontsize=11, fontweight='bold')
    for bar, m in zip(single_bars, best_singles):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f'{m:.1f}', ha='center', fontsize=9, fontstyle='italic',
                 alpha=0.7)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, fontsize=8)
    ax1.legend(fontsize=6, loc='upper right')
    ax1.set_ylabel('MAE (MPa)'); ax1.set_ylim(0, max(avgs)*1.6)
    ax1.set_title('V13 Results vs Baselines (Ensemble + Best Single)',
                  fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # ── Plot 2: Per-fold grouped bars ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(1, 6)
    w = 0.35
    for i, (n, col) in enumerate(zip(names, cols)):
        fold_vals = all_results[n]['folds']
        ax2.bar(x + (i - 0.5) * w, fold_vals, w, color=col, alpha=0.8,
                label=n + ' (ens)', edgecolor='white')
    ax2.axhline(95.99, color='#4CAF50', ls='-.', lw=1.5, label='V12A (95.99)')
    ax2.axhline(87.76, color='#F57F17', ls='--', lw=1.5, label='MODNet (87.76)')
    ax2.set_xlabel('Fold'); ax2.set_ylabel('MAE (MPa)')
    ax2.set_xticks(x); ax2.set_xticklabels([f'F{i}' for i in range(1,6)])
    ax2.set_title('Per-Fold Ensemble Breakdown', fontweight='bold')
    ax2.legend(fontsize=7); ax2.grid(axis='y', alpha=0.2)

    # ── Plot 3: Training/Val loss curves ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for cname, col in PAL.items():
        if cname not in all_hists: continue
        for fi, h in enumerate(all_hists[cname]):
            lb_tr = f'{cname} train' if fi == 0 else None
            lb_vl = f'{cname} val'   if fi == 0 else None
            ax3.plot(h['train'], alpha=0.3, lw=0.8, color=col, label=lb_tr)
            ax3.plot(h['val'],   alpha=0.7, lw=1.2, color=col, label=lb_vl,
                     linestyle='--')
    ax3.axhline(95.99, color='#4CAF50', ls='-.', lw=1.2, label='V12A (95.99)')
    ax3.axvline(200, color='#4CAF50', ls='--', lw=1.2, alpha=0.6, label='SWA start')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('MAE (MPa)')
    ax3.set_title('Training Curves (seed 0, all folds)', fontweight='bold')
    ax3.legend(fontsize=6, ncol=2); ax3.grid(alpha=0.2)
    ax3.set_ylim(0, 300)

    # ── Plot 4: Per-seed scatter / Confidence ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if all_conf_weights:
        for cn, fw_list in all_conf_weights.items():
            all_w = torch.cat(fw_list, dim=0)
            avg_w = all_w.mean(dim=0).numpy()
            steps = np.arange(1, len(avg_w)+1)
            ax4.bar(steps, avg_w, color=PAL.get(cn, '#E65100'), alpha=0.8,
                    label=f'{cn} avg confidence', edgecolor='white')
            std_w = all_w.std(dim=0).numpy()
            ax4.errorbar(steps, avg_w, yerr=std_w, fmt='none',
                        ecolor='#333', capsize=2, alpha=0.5)
        ax4.set_xlabel('Recursion Step')
        ax4.set_ylabel('Confidence Weight (softmax)')
        ax4.set_title('V13B: Where the Model Trusts Its Predictions',
                      fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(axis='y', alpha=0.2)
    else:
        # Show per-seed MAE scatter if no confidence model
        for i, (cn, col) in enumerate(zip(names, cols)):
            r = all_results[cn]
            seed_avgs = [r['per_seed_avgs'][s] for s in SEEDS]
            ax4.scatter(SEEDS, seed_avgs, s=80, c=col, alpha=0.8,
                       label=f'{cn} per-seed', zorder=5,
                       edgecolors='white', linewidth=1)
            ax4.axhline(r['avg'], color=col, ls='--', lw=1.5, alpha=0.6,
                       label=f'{cn} ensemble={r["avg"]:.2f}')
        ax4.axhline(95.99, color='#4CAF50', ls=':', lw=1, alpha=0.5, label='V12A')
        ax4.set_xlabel('Random Seed')
        ax4.set_ylabel('5-Fold Avg MAE (MPa)')
        ax4.set_title('Per-Seed vs Ensemble Performance', fontweight='bold')
        ax4.legend(fontsize=7); ax4.grid(alpha=0.2)

    fig.suptitle('TRM-MatSci V13 │ 2-Layer SA + Multi-Seed Ensemble │ matbench_steels',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.savefig('trm_results_v13.png', dpi=150, bbox_inches='tight')
    plt.close(fig); log.info("✓ Saved: trm_results_v13.png")


def save_summary(all_results, all_hists, all_conf_weights, total_s):
    # Prepare confidence info
    conf_info = {}
    for cn, fw_list in all_conf_weights.items():
        all_w = torch.cat(fw_list, dim=0)
        conf_info[cn] = {
            'avg_weights': all_w.mean(dim=0).numpy().round(4).tolist(),
            'avg_peak_step': float(all_w.argmax(dim=1).float().mean().item() + 1),
        }

    s = {
        'version': 'V13', 'task': 'matbench_steels',
        'strategy': '2-Layer SA + Multi-Seed Ensemble',
        'seeds': SEEDS,
        'n_seeds': N_SEEDS,
        'total_min': round(total_s/60, 1),
        'models': {},
        'confidence': conf_info,
    }
    for n, r in all_results.items():
        s['models'][n] = {
            'ensemble_avg': round(r['avg'], 4),
            'ensemble_std': round(r['std'], 4),
            'ensemble_folds': [round(x, 4) for x in r['folds']],
            'params': r['params'],
            'best_single_seed': r['best_single_seed'],
            'best_single_mae': round(r['best_single_mae'], 4),
            'per_seed_avgs': {str(k): round(v, 4) for k, v in r['per_seed_avgs'].items()},
        }

    with open('trm_models_v13/summary_v13.json', 'w') as f:
        json.dump(s, f, indent=2, default=str)
    log.info("✓ Saved: summary_v13.json")


if __name__ == '__main__':
    results = run_benchmark()
    shutil.make_archive("trm_v13_all", "zip", "trm_models_v13")
    log.info("✓ Created trm_v13_all.zip")

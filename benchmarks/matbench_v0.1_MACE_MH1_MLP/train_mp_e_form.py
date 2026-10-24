import warnings
warnings.filterwarnings("ignore")

import os
import time
import numpy as np
import pickle
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_NAME = "matbench_mp_e_form"
np.random.seed(SEED)
torch.manual_seed(SEED)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Loading fold indices...")
fold_info = np.load(os.path.join(SAVE_DIR, f"folds_{TASK_NAME}.npz"))
n_total = int(fold_info["n_total"][0])
log(f"Dataset: {n_total} structures")

log("Loading frozen descriptors...")
all_descs = np.load(os.path.join(SAVE_DIR, f"desc_{TASK_NAME}.npz"))["descriptors"].astype(np.float32)
n_desc = all_descs.shape[1]
log(f"  Frozen descriptors: {all_descs.shape}")

log("Loading MACE calculator features...")
X_calc = np.load(os.path.join(SAVE_DIR, f"calc_{TASK_NAME}.npz"))["calc_features"].astype(np.float32)
n_calc = X_calc.shape[1]
log(f"  Calculator features: {X_calc.shape}")

log("Loading Magpie composition features...")
with open(os.path.join(SAVE_DIR, f"magpie_{TASK_NAME}.pkl"), "rb") as f:
    X_magpie = pickle.load(f).astype(np.float32)
n_magpie = X_magpie.shape[1]
log(f"  Magpie features: {X_magpie.shape}")

n_feat = n_desc + n_calc + n_magpie
X_all = np.empty((n_total, n_feat), dtype=np.float32)
X_all[:, :n_desc] = all_descs; del all_descs
X_all[:, n_desc:n_desc+n_calc] = X_calc; del X_calc
X_all[:, n_desc+n_calc:] = X_magpie; del X_magpie
gc.collect()

log(f"Combined: {X_all.shape} ({n_desc} desc + {n_calc} calc + {n_magpie} magpie)")


class PropertyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_fold(X_train, y_train, X_test, fold_idx, epochs=200, lr=1e-3,
                   batch_size=512, hidden_dim=256, dropout=0.1):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tr_idx, val_idx = train_test_split(
        range(len(X_train)), test_size=0.1, random_state=SEED)
    log(f"  Split: {len(tr_idx)} train, {len(val_idx)} val, {len(X_test)} test")

    scaler = StandardScaler()
    X_tr_np = scaler.fit_transform(X_train[tr_idx])
    X_val_np = scaler.transform(X_train[val_idx])
    X_te_np = scaler.transform(X_test)

    X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
    y_tr = torch.tensor(y_train[tr_idx], dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    y_val = torch.tensor(y_train[val_idx], dtype=torch.float32)
    X_te = torch.tensor(X_te_np, dtype=torch.float32)
    del X_tr_np, X_val_np, X_te_np; gc.collect()

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    model = PropertyMLP(input_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  MLP: {n_params:,} params, input_dim={X_train.shape[1]}, hidden={hidden_dim}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.L1Loss()

    best_val_mae = float('inf')
    best_state = None
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vp = torch.cat([model(xb.to(DEVICE)).cpu() for xb, _ in val_loader])
            val_mae = torch.mean(torch.abs(vp - y_val)).item()

        improved = ""
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            improved = " *"

        if epoch % 20 == 0 or improved:
            log(f"  Epoch {epoch:3d}/{epochs}: train={np.mean(epoch_losses):.4f} "
                f"val={val_mae:.4f} best={best_val_mae:.4f}{improved}")

    log(f"  Best val_MAE={best_val_mae:.4f} in {time.time()-t0:.1f}s")

    model.load_state_dict(best_state)
    model.to(DEVICE).eval()
    with torch.no_grad():
        preds = model(X_te.to(DEVICE)).cpu().numpy()
    return preds, best_val_mae


log("")
log("=" * 60)
log(f"TRAINING MLP - 5 FOLDS ({TASK_NAME})")
log("=" * 60)

all_preds = {}
fold_maes = []

for fold in range(5):
    log(f"\n--- Fold {fold} ---")
    tr_pos = fold_info[f"tr_pos_{fold}"]
    te_pos = fold_info[f"te_pos_{fold}"]
    y_tr = fold_info[f"y_tr_{fold}"]
    y_te = fold_info[f"y_te_{fold}"]

    X_tr = X_all[tr_pos]
    X_te = X_all[te_pos]

    imp = SimpleImputer(strategy="median")
    X_tr = np.nan_to_num(imp.fit_transform(X_tr), nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(imp.transform(X_te), nan=0.0, posinf=0.0, neginf=0.0)

    preds, _ = train_mlp_fold(X_tr, y_tr, X_te, fold)

    mae = mean_absolute_error(y_te, preds)
    fold_maes.append(mae)
    all_preds[f"preds_{fold}"] = preds
    all_preds[f"te_pos_{fold}"] = te_pos
    log(f"Fold {fold}: test MAE = {mae:.4f}")

    del X_tr, X_te; gc.collect()

mean_mae = np.mean(fold_maes)
std_mae = np.std(fold_maes)

log("")
log("=" * 60)
log(f"RESULTS ({TASK_NAME})")
log("=" * 60)
log(f"Per-fold MAE: {[f'{m:.4f}' for m in fold_maes]}")
log(f"Mean MAE: {mean_mae:.4f} +/- {std_mae:.4f}")

out_path = os.path.join(SAVE_DIR, f"preds_{TASK_NAME}.npz")
np.savez(out_path, **all_preds)
log(f"Predictions saved to {out_path}")

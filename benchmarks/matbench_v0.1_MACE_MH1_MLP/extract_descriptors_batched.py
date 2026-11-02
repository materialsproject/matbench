import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 32
NUM_THREADS = 8

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Loading MACE-MH-1...")
from mace.calculators import mace_mp
calc = mace_mp(model="mh-1", default_dtype="float32", device=DEVICE, head="omat_pbe")
model = calc.models[0]
model.eval()

from mace.tools.utils import AtomicNumberTable
z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
r_max = model.r_max.item()
num_interactions = int(model.num_interactions)

from e3nn import o3
irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
l_max = irreps_out.lmax
num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
DESC_DIM = num_invariant_features * num_interactions

log(f"Model loaded. Device={DEVICE}, r_max={r_max}, "
    f"num_interactions={num_interactions}, invariant_dims={DESC_DIM}")

from mace.data import AtomicData
from mace.data.utils import config_from_atoms, KeySpecification
from mace.tools.torch_geometric.dataloader import DataLoader
from mace.modules.utils import extract_invariant
from pymatgen.io.ase import AseAtomsAdaptor

adaptor = AseAtomsAdaptor()
key_spec = KeySpecification()

N_CALC_FEATS = 19


def struct_to_atomic_data(struct):
    atoms = adaptor.get_atoms(struct, msonable=False)
    config = config_from_atoms(atoms, key_specification=key_spec)
    return AtomicData.from_config(
        config, z_table=z_table, cutoff=r_max, heads=["omat_pbe"],
    )


def calc_feats_from_output(output, batch_dict, ptr, i):
    start, end = ptr[i].item(), ptr[i + 1].item()
    n_atoms = end - start

    energy = output["energy"][i].item() if "energy" in output else np.nan
    epa = energy / n_atoms if not np.isnan(energy) else np.nan

    node_e = output.get("node_energy", None)
    if node_e is not None:
        ae = node_e[start:end].detach().cpu().numpy().flatten()
        ae_mean, ae_std = np.mean(ae), np.std(ae)
        ae_min, ae_max = np.min(ae), np.max(ae)
        ae_range = ae_max - ae_min
    else:
        ae_mean = ae_std = ae_min = ae_max = ae_range = np.nan

    forces = output.get("forces", None)
    if forces is not None:
        f = forces[start:end].detach().cpu().numpy()
        fm = np.linalg.norm(f, axis=1)
        fm_mean, fm_std, fm_max = np.mean(fm), np.std(fm), np.max(fm)
    else:
        fm_mean = fm_std = fm_max = np.nan

    stress = output.get("stress", None)
    if stress is not None and stress.numel() > 0:
        s = stress[i].detach().cpu().numpy().flatten()
        if len(s) >= 6:
            hydro = (s[0] + s[1] + s[2]) / 3.0
            max_abs = np.max(np.abs(s[:6]))
            s_feats = list(s[:6]) + [hydro, max_abs]
        else:
            s_feats = [np.nan] * 8
    else:
        s_feats = [np.nan] * 8

    return [energy, epa, n_atoms,
            ae_mean, ae_std, ae_min, ae_max, ae_range,
            fm_mean, fm_std, fm_max] + s_feats


def extract_batch(task_name, structures, save_path):
    total = len(structures)
    descs = np.zeros((total, DESC_DIM), dtype=np.float32)
    calc_feats = np.full((total, N_CALC_FEATS), np.nan, dtype=np.float32)
    failed = 0
    t0 = time.time()

    CHUNK = BATCH_SIZE * 4

    for chunk_start in range(0, total, CHUNK):
        chunk_end = min(chunk_start + CHUNK, total)
        chunk_structs = structures[chunk_start:chunk_end]

        valid_data = []
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = {executor.submit(struct_to_atomic_data, s): i
                       for i, s in enumerate(chunk_structs)}
            for future in futures:
                local_idx = futures[future]
                global_idx = chunk_start + local_idx
                try:
                    data = future.result()
                    valid_data.append((global_idx, data))
                except Exception:
                    failed += 1

        if not valid_data:
            continue

        valid_data.sort(key=lambda x: x[0])
        indices = [x[0] for x in valid_data]
        data_list = [x[1] for x in valid_data]

        loader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        desc_offset = 0
        for batch in loader:
            batch_size_actual = len(batch.ptr) - 1
            try:
                batch = batch.to(DEVICE)
                batch_dict = batch.to_dict()
                batch_dict["positions"].requires_grad_(True)
                output = model(batch_dict)
                node_feats = output["node_feats"].detach()

                inv_feats = extract_invariant(
                    node_feats,
                    num_layers=num_interactions,
                    num_features=num_invariant_features,
                    l_max=l_max,
                )

                ptr = batch.ptr
                for i in range(len(ptr) - 1):
                    start, end = ptr[i].item(), ptr[i + 1].item()
                    mean_desc = inv_feats[start:end].mean(dim=0).cpu().numpy()
                    global_idx = indices[desc_offset]
                    descs[global_idx] = mean_desc
                    try:
                        calc_feats[global_idx] = calc_feats_from_output(output, batch_dict, ptr, i)
                    except Exception:
                        pass
                    desc_offset += 1
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                for j in range(batch_size_actual):
                    try:
                        single = data_list[desc_offset]
                        sl = DataLoader([single], batch_size=1, shuffle=False)
                        sb = next(iter(sl)).to(DEVICE)
                        sd = sb.to_dict()
                        sd["positions"].requires_grad_(True)
                        out = model(sd)
                        nf = out["node_feats"].detach()
                        inv = extract_invariant(nf, num_layers=num_interactions,
                                                num_features=num_invariant_features, l_max=l_max)
                        descs[indices[desc_offset]] = inv.mean(dim=0).cpu().numpy()
                        try:
                            calc_feats[indices[desc_offset]] = calc_feats_from_output(
                                out, sd, sb.ptr, 0)
                        except Exception:
                            pass
                    except Exception:
                        failed += 1
                        torch.cuda.empty_cache()
                    desc_offset += 1

        done = min(chunk_end, total)
        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (total - done) / rate if rate > 0 else 0
        log(f"  [{task_name}] {done}/{total} ({rate:.0f}/s, "
            f"ETA {eta:.0f}s, {failed} failures)")

    calc_path = save_path.replace("desc_", "calc_")
    np.savez(save_path, descriptors=descs)
    np.savez(calc_path, calc_features=calc_feats)
    log(f"[{task_name}] Saved calc features to {calc_path}")
    total_time = time.time() - t0
    log(f"[{task_name}] Saved {save_path} ({total} structs, {failed} failures, "
        f"{total_time:.0f}s total, {total/total_time:.1f} structs/s)")


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("tasks", nargs="*",
                    default=["matbench_mp_e_form", "matbench_mp_gap", "matbench_mp_is_metal"])
parser.add_argument("--smoke", type=int, default=0)
args = parser.parse_args()

import pickle
from matbench.bench import MatbenchBenchmark


def load_structures(task_name):
    pkl_path = os.path.join(SAVE_DIR, f"structs_{task_name}.pkl")
    if os.path.exists(pkl_path):
        log(f"[{task_name}] Loading cached structures from pickle...")
        t = time.time()
        with open(pkl_path, "rb") as f:
            structures = pickle.load(f)
        log(f"[{task_name}] Loaded {len(structures)} structures in {time.time()-t:.1f}s")
        return structures

    log(f"[{task_name}] Loading dataset from matbench (first time, slow)...")
    mb = MatbenchBenchmark(autoload=False, subset=[task_name])
    task = list(mb.tasks)[0]
    task.load()
    structures = list(task.df["structure"])
    with open(pkl_path, "wb") as f:
        pickle.dump(structures, f)
    log(f"[{task_name}] {len(structures)} structures cached to {pkl_path}")
    return structures


for task_name in args.tasks:
    suffix = "_smoke" if args.smoke else ""
    cache_path = os.path.join(SAVE_DIR, f"desc_{task_name}{suffix}.npz")
    if os.path.exists(cache_path) and not args.smoke:
        log(f"[{task_name}] Already cached at {cache_path}, skipping.")
        continue

    structures = load_structures(task_name)
    if args.smoke:
        structures = structures[:args.smoke]
        log(f"[{task_name}] SMOKE TEST: {len(structures)} structures")

    extract_batch(task_name, structures, cache_path)

log("All done.")

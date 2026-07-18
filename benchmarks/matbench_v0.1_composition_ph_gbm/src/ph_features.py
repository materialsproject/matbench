# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pandas",
#   "persim",
#   "pyarrow",
#   "pymatgen",
#   "ripser",
# ]
# ///

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import persim
import pymatgen
import ripser as ripser_package
from pymatgen.core import Structure
from ripser import ripser


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "data" / "matbench_log_gvrh.json.gz"
DEFAULT_CACHE = HERE / "run_outputs" / "matbench_gvrh" / "ph_features.parquet"
DEFAULT_CONFIG = HERE / "run_outputs" / "matbench_gvrh" / "ph_featurizer_config.json"

DATASET_NAME = "matbench_log_gvrh"
EXPECTED_ROWS = 10_987
DEFAULT_R_CUT = 6.0
DEFAULT_MAX_POINTS = 1_000
DEFAULT_HIST_BINS = 8
PH_DIMENSIONS = (0, 1)
CHECKPOINT_EVERY = 500


def mbids(n_rows: int) -> list[str]:
    width = len(str(n_rows))
    return [f"mb-log-gvrh-{i + 1:0{width}d}" for i in range(n_rows)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_structure_dicts(dataset_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with gzip.open(dataset_path, "rt") as handle:
        raw = json.load(handle)
    if raw["columns"] != ["structure", "log10(G_VRH)"]:
        raise RuntimeError(f"unexpected columns: {raw['columns']}")
    if len(raw["data"]) != EXPECTED_ROWS:
        raise RuntimeError(f"unexpected row count: {len(raw['data'])}")
    return mbids(len(raw["data"])), [row[0] for row in raw["data"]]


def feature_names(hist_bins: int = DEFAULT_HIST_BINS) -> list[str]:
    names: list[str] = []
    for dim in PH_DIMENSIONS:
        prefix = f"ph{dim}"
        names.extend(
            [
                f"{prefix}_finite_count",
                f"{prefix}_persistence_total",
                f"{prefix}_persistence_max",
                f"{prefix}_persistence_mean",
                f"{prefix}_persistence_std",
                f"{prefix}_birth_mean",
                f"{prefix}_birth_std",
                f"{prefix}_birth_q25",
                f"{prefix}_birth_q50",
                f"{prefix}_birth_q75",
                f"{prefix}_death_mean",
                f"{prefix}_death_std",
                f"{prefix}_death_q25",
                f"{prefix}_death_q50",
                f"{prefix}_death_q75",
            ]
        )
        names.extend(f"{prefix}_persistence_hist_{i}" for i in range(hist_bins))
    names.extend(
        [
            "geometry_unit_cell_atom_count",
            "geometry_unit_cell_volume_a3",
            "geometry_atom_density_per_a3",
            "geometry_point_cloud_count",
            "geometry_expansion_fallback_used",
        ]
    )
    return names


def _coordinate_key(coords: np.ndarray) -> tuple[float, float, float]:
    rounded = np.round(np.asarray(coords, dtype=float), decimals=8)
    return float(rounded[0]), float(rounded[1]), float(rounded[2])


def periodic_point_cloud(
    structure: Structure,
    *,
    r_cut: float,
    max_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    # get_all_neighbors enumerates every periodic image within r_cut of every
    # central-cell site. The union therefore gives each central atom complete
    # neighbor coverage without selecting atoms using the target property.
    points = {_coordinate_key(site.coords) for site in structure}
    for neighbors in structure.get_all_neighbors(r_cut, include_index=True):
        for neighbor in neighbors:
            points.add(_coordinate_key(neighbor.coords))
    cloud = np.asarray(sorted(points), dtype=float)
    initial_count = int(len(cloud))
    fallback_used = False
    hard_cap_used = False

    if len(cloud) > max_points:
        fallback_used = True
        supercell = structure.copy()
        supercell.make_supercell([2, 2, 2])
        cloud = np.asarray(sorted({_coordinate_key(site.coords) for site in supercell}), dtype=float)

    if len(cloud) > max_points:
        # Large conventional cells can exceed the cap even at 2x2x2. Retain
        # the closest points to the central-cell centroid deterministically.
        hard_cap_used = True
        centroid = np.mean(structure.cart_coords, axis=0)
        distances = np.linalg.norm(cloud - centroid, axis=1)
        order = np.lexsort((cloud[:, 2], cloud[:, 1], cloud[:, 0], distances))
        cloud = cloud[order[:max_points]]

    return cloud, {
        "coverage_cloud_count": initial_count,
        "point_cloud_count": int(len(cloud)),
        "expansion_fallback_used": fallback_used,
        "hard_cap_used": hard_cap_used,
    }


def diagram_features(
    diagram: np.ndarray,
    *,
    dim: int,
    r_cut: float,
    hist_bins: int,
) -> dict[str, float]:
    prefix = f"ph{dim}"
    if diagram.size == 0:
        finite = np.empty((0, 2), dtype=float)
    else:
        finite = np.asarray(diagram, dtype=float)
        finite = finite[np.isfinite(finite[:, 0]) & np.isfinite(finite[:, 1])]

    output = {name: 0.0 for name in feature_names(hist_bins) if name.startswith(prefix)}
    output[f"{prefix}_finite_count"] = float(len(finite))
    if len(finite) == 0:
        return output

    births = finite[:, 0]
    deaths = finite[:, 1]
    persistence = np.maximum(0.0, deaths - births)
    output.update(
        {
            f"{prefix}_persistence_total": float(np.sum(persistence)),
            f"{prefix}_persistence_max": float(np.max(persistence)),
            f"{prefix}_persistence_mean": float(np.mean(persistence)),
            f"{prefix}_persistence_std": float(np.std(persistence)),
            f"{prefix}_birth_mean": float(np.mean(births)),
            f"{prefix}_birth_std": float(np.std(births)),
            f"{prefix}_birth_q25": float(np.quantile(births, 0.25)),
            f"{prefix}_birth_q50": float(np.quantile(births, 0.50)),
            f"{prefix}_birth_q75": float(np.quantile(births, 0.75)),
            f"{prefix}_death_mean": float(np.mean(deaths)),
            f"{prefix}_death_std": float(np.std(deaths)),
            f"{prefix}_death_q25": float(np.quantile(deaths, 0.25)),
            f"{prefix}_death_q50": float(np.quantile(deaths, 0.50)),
            f"{prefix}_death_q75": float(np.quantile(deaths, 0.75)),
        }
    )
    histogram, _ = np.histogram(persistence, bins=hist_bins, range=(0.0, r_cut))
    for i, value in enumerate(histogram):
        output[f"{prefix}_persistence_hist_{i}"] = float(value)
    return output


def featurize_structure(
    structure: Structure,
    *,
    r_cut: float,
    max_points: int,
    hist_bins: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    cloud, cloud_info = periodic_point_cloud(
        structure,
        r_cut=r_cut,
        max_points=max_points,
    )
    diagrams = ripser(cloud, maxdim=max(PH_DIMENSIONS), thresh=r_cut)["dgms"]
    output: dict[str, float] = {}
    for dim in PH_DIMENSIONS:
        output.update(
            diagram_features(
                diagrams[dim],
                dim=dim,
                r_cut=r_cut,
                hist_bins=hist_bins,
            )
        )
    output.update(
        {
            "geometry_unit_cell_atom_count": float(len(structure)),
            "geometry_unit_cell_volume_a3": float(structure.volume),
            "geometry_atom_density_per_a3": float(len(structure) / structure.volume),
            "geometry_point_cloud_count": float(len(cloud)),
            "geometry_expansion_fallback_used": float(cloud_info["expansion_fallback_used"]),
        }
    )
    expected = feature_names(hist_bins)
    if list(output) != expected:
        raise RuntimeError("feature order does not match declared feature names")
    values = np.asarray(list(output.values()), dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("non-finite persistent-homology feature")
    return output, cloud_info


def base_config(
    *,
    dataset_path: Path,
    r_cut: float,
    max_points: int,
    hist_bins: int,
) -> dict[str, Any]:
    return {
        "dataset": DATASET_NAME,
        "dataset_path": str(dataset_path.relative_to(HERE)),
        "dataset_sha256": sha256_file(dataset_path),
        "library": "ripser + persim",
        "ripser_version": ripser_package.__version__,
        "persim_version": persim.__version__,
        "pymatgen_version": getattr(pymatgen, "__version__", "unknown"),
        "filtration": "Vietoris-Rips",
        "diagram_dimensions": list(PH_DIMENSIONS),
        "r_cut_angstrom": r_cut,
        "ripser_thresh_angstrom": r_cut,
        "max_points": max_points,
        "oversize_fallback": "deterministic 2x2x2 expansion, then nearest-to-central-centroid hard cap if still needed",
        "histogram_bins": hist_bins,
        "histogram_range": [0.0, r_cut],
        "feature_names": feature_names(hist_bins),
        "n_features": len(feature_names(hist_bins)),
        "label_access": "none; structure only",
        "random_sampling": "none",
    }


def run_probe(
    structure_dicts: list[dict[str, Any]],
    *,
    count: int,
    r_cut: float,
    max_points: int,
    hist_bins: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    point_counts: list[int] = []
    fallback_count = 0
    hard_cap_count = 0
    for structure_dict in structure_dicts[:count]:
        _, info = featurize_structure(
            Structure.from_dict(structure_dict),
            r_cut=r_cut,
            max_points=max_points,
            hist_bins=hist_bins,
        )
        point_counts.append(info["point_cloud_count"])
        fallback_count += int(info["expansion_fallback_used"])
        hard_cap_count += int(info["hard_cap_used"])
    elapsed = time.perf_counter() - started
    return {
        "probe_rows": count,
        "elapsed_seconds": elapsed,
        "seconds_per_structure": elapsed / count,
        "projected_full_seconds": elapsed / count * len(structure_dicts),
        "projected_full_hours": elapsed / count * len(structure_dicts) / 3600.0,
        "point_count_min": min(point_counts),
        "point_count_mean": float(np.mean(point_counts)),
        "point_count_max": max(point_counts),
        "expansion_fallback_count": fallback_count,
        "hard_cap_count": hard_cap_count,
        "r_cut_angstrom": r_cut,
        "diagram_dimensions": list(PH_DIMENSIONS),
    }


def write_cache(records: list[dict[str, Any]], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    if frame["mbid"].duplicated().any():
        raise RuntimeError("duplicate mbids in feature cache")
    frame.to_parquet(cache_path, index=False)


def compute_cache(
    ids: list[str],
    structure_dicts: list[dict[str, Any]],
    *,
    dataset_path: Path,
    cache_path: Path,
    config_path: Path,
    r_cut: float,
    max_points: int,
    hist_bins: int,
) -> dict[str, Any]:
    config = base_config(
        dataset_path=dataset_path,
        r_cut=r_cut,
        max_points=max_points,
        hist_bins=hist_bins,
    )
    records: list[dict[str, Any]] = []
    previous_seconds = 0.0
    hard_cap_count = 0
    if cache_path.exists():
        if not config_path.exists():
            raise RuntimeError("feature cache exists without config JSON")
        existing_config = json.loads(config_path.read_text())
        for key in (
            "dataset_sha256",
            "library",
            "diagram_dimensions",
            "r_cut_angstrom",
            "max_points",
            "histogram_bins",
            "feature_names",
        ):
            if existing_config[key] != config[key]:
                raise RuntimeError(f"cached config mismatch for {key}")
        records = pd.read_parquet(cache_path).to_dict(orient="records")
        if [record["mbid"] for record in records] != ids[: len(records)]:
            raise RuntimeError("cache mbids are not a valid dataset prefix")
        previous_seconds = float(existing_config.get("total_featurization_seconds", 0.0))
        hard_cap_count = int(existing_config.get("hard_cap_count", 0))

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.update(
        {
            "completed_rows": len(records),
            "cache_path": str(cache_path.relative_to(HERE)),
            "total_featurization_seconds": previous_seconds,
            "hard_cap_count": hard_cap_count,
        }
    )
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    started = time.perf_counter()
    point_counts = [int(record["geometry_point_cloud_count"]) for record in records]
    fallback_count = sum(int(record["geometry_expansion_fallback_used"]) for record in records)
    start_index = len(records)
    print(f"resuming persistent-homology features at {start_index}/{len(ids)}", flush=True)
    for i in range(start_index, len(ids)):
        row_started = time.perf_counter()
        features, info = featurize_structure(
            Structure.from_dict(structure_dicts[i]),
            r_cut=r_cut,
            max_points=max_points,
            hist_bins=hist_bins,
        )
        records.append({"mbid": ids[i], **features})
        point_counts.append(info["point_cloud_count"])
        fallback_count += int(info["expansion_fallback_used"])
        hard_cap_count += int(info["hard_cap_used"])
        completed = i + 1
        if completed % CHECKPOINT_EVERY == 0 or completed == len(ids):
            write_cache(records, cache_path)
            total_elapsed = time.perf_counter() - started
            config.update(
                {
                    "completed_rows": completed,
                    "cache_sha256": sha256_file(cache_path),
                    "expansion_fallback_count": fallback_count,
                    "hard_cap_count": hard_cap_count,
                    "last_session_start_index": start_index,
                    "last_session_seconds": total_elapsed,
                    "total_featurization_seconds": previous_seconds + total_elapsed,
                }
            )
            config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            print(
                f"persistent-homology progress {completed}/{len(ids)}; "
                f"last_seconds={time.perf_counter() - row_started:.3f}; "
                f"session_seconds={total_elapsed:.1f}",
                flush=True,
            )

    session_elapsed = time.perf_counter() - started
    config.update(
        {
            "completed_rows": len(records),
            "cache_path": str(cache_path.relative_to(HERE)),
            "cache_sha256": sha256_file(cache_path),
            "point_count_min": int(min(point_counts)),
            "point_count_mean": float(np.mean(point_counts)),
            "point_count_max": int(max(point_counts)),
            "expansion_fallback_count": fallback_count,
            "hard_cap_count": hard_cap_count,
            "last_session_start_index": start_index,
            "last_session_seconds": session_elapsed,
            "total_featurization_seconds": previous_seconds + session_elapsed,
        }
    )
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structure-only persistent homology features")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--r-cut", type=float, default=DEFAULT_R_CUT)
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS)
    parser.add_argument("--hist-bins", type=int, default=DEFAULT_HIST_BINS)
    parser.add_argument("--probe", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids, structure_dicts = load_structure_dicts(args.dataset)
    if args.probe:
        if args.probe < 1 or args.probe > len(ids):
            raise ValueError("--probe must be between 1 and the dataset row count")
        result = run_probe(
            structure_dicts,
            count=args.probe,
            r_cut=args.r_cut,
            max_points=args.max_points,
            hist_bins=args.hist_bins,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    config = compute_cache(
        ids,
        structure_dicts,
        dataset_path=args.dataset,
        cache_path=args.cache,
        config_path=args.config,
        r_cut=args.r_cut,
        max_points=args.max_points,
        hist_bins=args.hist_bins,
    )
    print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

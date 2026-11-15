"""Original deterministic structure-only descriptor from PR #363.

The reproduction driver owns input/code/version-keyed caching and reports
fallback/cap counts. No target values enter this module.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from pymatgen.core import Structure
from ripser import ripser

DEFAULT_HIST_BINS = 8
PH_DIMENSIONS = (0, 1)


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
        cloud = np.asarray(
            sorted({_coordinate_key(site.coords) for site in supercell}), dtype=float
        )

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

    output = {
        name: 0.0 for name in feature_names(hist_bins) if name.startswith(prefix)
    }
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
            "geometry_expansion_fallback_used": float(
                cloud_info["expansion_fallback_used"]
            ),
        }
    )
    expected = feature_names(hist_bins)
    if list(output) != expected:
        raise RuntimeError("feature order does not match declared feature names")
    values = np.asarray(list(output.values()), dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("non-finite persistent-homology feature")
    return output, cloud_info

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

from __future__ import annotations

import math
from typing import Any

from composition_features import parse_formula
from element_table_generated import ELEMENTS, ELEMENT_TABLE, PROPERTY_NAMES


FRACTION_FEATURE_NAMES = tuple(f"frac_{element}" for element in ELEMENTS)
GLOBAL_STAT_NAMES = (
    "n_elements",
    "total_atoms",
    "max_fraction",
    "composition_entropy",
    "other_fraction",
)
PROPERTY_STAT_NAMES = ("mean", "std", "min", "max", "range")
PROPERTY_FEATURE_NAMES = tuple(
    feature_name
    for property_name in PROPERTY_NAMES
    for feature_name in (
        *(f"{stat_name}_{property_name}" for stat_name in PROPERTY_STAT_NAMES),
        f"missing_{property_name}_fraction",
    )
)
FEATURE_NAMES = FRACTION_FEATURE_NAMES + GLOBAL_STAT_NAMES + PROPERTY_FEATURE_NAMES


def element_fractions(formula: str) -> dict[str, float]:
    counts = parse_formula(formula)
    total = sum(counts.values())
    if total <= 0:
        raise ValueError(f"formula has non-positive atom total: {formula}")
    return {element: amount / total for element, amount in counts.items()}


def weighted_mean(values: list[float], weights: list[float]) -> float:
    return sum(value * weight for value, weight in zip(values, weights))


def weighted_std(values: list[float], weights: list[float], mean: float) -> float:
    variance = sum(
        weight * (value - mean) ** 2 for value, weight in zip(values, weights)
    )
    return math.sqrt(max(0.0, variance))


def _property_stats(
    fractions: dict[str, float],
    property_name: str,
) -> dict[str, float]:
    values: list[float] = []
    weights_raw: list[float] = []
    missing_fraction = 0.0
    for element, fraction in fractions.items():
        row: dict[str, Any] | None = ELEMENT_TABLE.get(element)
        value = row.get(property_name) if row else None
        if value is None:
            missing_fraction += fraction
            continue
        values.append(float(value))
        weights_raw.append(fraction)

    out = {
        f"mean_{property_name}": 0.0,
        f"std_{property_name}": 0.0,
        f"min_{property_name}": 0.0,
        f"max_{property_name}": 0.0,
        f"range_{property_name}": 0.0,
        f"missing_{property_name}_fraction": float(missing_fraction),
    }
    known_weight = sum(weights_raw)
    if known_weight <= 0:
        return out

    weights = [weight / known_weight for weight in weights_raw]
    mean = weighted_mean(values, weights)
    min_value = min(values)
    max_value = max(values)
    out[f"mean_{property_name}"] = float(mean)
    out[f"std_{property_name}"] = float(weighted_std(values, weights, mean))
    out[f"min_{property_name}"] = float(min_value)
    out[f"max_{property_name}"] = float(max_value)
    out[f"range_{property_name}"] = float(max_value - min_value)
    return out


def composition_features(formula: str) -> dict[str, float]:
    counts = parse_formula(formula)
    total = sum(counts.values())
    if total <= 0:
        raise ValueError(f"formula has non-positive atom total: {formula}")

    fractions = {element: amount / total for element, amount in counts.items()}
    out = {name: 0.0 for name in FEATURE_NAMES}
    for element in ELEMENTS:
        out[f"frac_{element}"] = fractions.get(element, 0.0)

    out["n_elements"] = float(len(fractions))
    out["total_atoms"] = float(total)
    out["max_fraction"] = float(max(fractions.values()))
    out["composition_entropy"] = float(
        -sum(fraction * math.log(fraction) for fraction in fractions.values())
    )
    out["other_fraction"] = float(
        sum(
            fraction
            for element, fraction in fractions.items()
            if element not in ELEMENT_TABLE
        )
    )

    for property_name in PROPERTY_NAMES:
        out.update(_property_stats(fractions, property_name))

    return out


def feature_vector(formula: str) -> list[float]:
    features = composition_features(formula)
    return [features[name] for name in FEATURE_NAMES]

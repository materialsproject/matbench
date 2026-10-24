# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ElementProperty:
    z: int
    period: int
    group: int
    electronegativity: float


# Compact local reference table, hand-entered for common WBM-style compositions.
# It is not a downloaded dataset. Electronegativity values are Pauling-scale
# reference constants rounded to two decimals for lightweight featurization;
# noble gases without defined Pauling values use 0.0 as a neutral placeholder.
ELEMENT_PROPERTIES: dict[str, ElementProperty] = {
    "H": ElementProperty(1, 1, 1, 2.20),
    "Li": ElementProperty(3, 2, 1, 0.98),
    "Be": ElementProperty(4, 2, 2, 1.57),
    "B": ElementProperty(5, 2, 13, 2.04),
    "C": ElementProperty(6, 2, 14, 2.55),
    "N": ElementProperty(7, 2, 15, 3.04),
    "O": ElementProperty(8, 2, 16, 3.44),
    "F": ElementProperty(9, 2, 17, 3.98),
    "Ne": ElementProperty(10, 2, 18, 0.00),
    "Na": ElementProperty(11, 3, 1, 0.93),
    "Mg": ElementProperty(12, 3, 2, 1.31),
    "Al": ElementProperty(13, 3, 13, 1.61),
    "Si": ElementProperty(14, 3, 14, 1.90),
    "P": ElementProperty(15, 3, 15, 2.19),
    "S": ElementProperty(16, 3, 16, 2.58),
    "Cl": ElementProperty(17, 3, 17, 3.16),
    "Ar": ElementProperty(18, 3, 18, 0.00),
    "K": ElementProperty(19, 4, 1, 0.82),
    "Ca": ElementProperty(20, 4, 2, 1.00),
    "Sc": ElementProperty(21, 4, 3, 1.36),
    "Ti": ElementProperty(22, 4, 4, 1.54),
    "V": ElementProperty(23, 4, 5, 1.63),
    "Cr": ElementProperty(24, 4, 6, 1.66),
    "Mn": ElementProperty(25, 4, 7, 1.55),
    "Fe": ElementProperty(26, 4, 8, 1.83),
    "Co": ElementProperty(27, 4, 9, 1.88),
    "Ni": ElementProperty(28, 4, 10, 1.91),
    "Cu": ElementProperty(29, 4, 11, 1.90),
    "Zn": ElementProperty(30, 4, 12, 1.65),
    "Ga": ElementProperty(31, 4, 13, 1.81),
    "Ge": ElementProperty(32, 4, 14, 2.01),
    "As": ElementProperty(33, 4, 15, 2.18),
    "Se": ElementProperty(34, 4, 16, 2.55),
    "Br": ElementProperty(35, 4, 17, 2.96),
    "Rb": ElementProperty(37, 5, 1, 0.82),
    "Sr": ElementProperty(38, 5, 2, 0.95),
    "Y": ElementProperty(39, 5, 3, 1.22),
    "Zr": ElementProperty(40, 5, 4, 1.33),
    "Nb": ElementProperty(41, 5, 5, 1.60),
    "Mo": ElementProperty(42, 5, 6, 2.16),
    "Ru": ElementProperty(44, 5, 8, 2.20),
    "Rh": ElementProperty(45, 5, 9, 2.28),
    "Pd": ElementProperty(46, 5, 10, 2.20),
    "Ag": ElementProperty(47, 5, 11, 1.93),
    "Cd": ElementProperty(48, 5, 12, 1.69),
    "In": ElementProperty(49, 5, 13, 1.78),
    "Sn": ElementProperty(50, 5, 14, 1.96),
    "Sb": ElementProperty(51, 5, 15, 2.05),
    "Te": ElementProperty(52, 5, 16, 2.10),
    "I": ElementProperty(53, 5, 17, 2.66),
    "Cs": ElementProperty(55, 6, 1, 0.79),
    "Ba": ElementProperty(56, 6, 2, 0.89),
    "La": ElementProperty(57, 6, 3, 1.10),
    "Hf": ElementProperty(72, 6, 4, 1.30),
    "Ta": ElementProperty(73, 6, 5, 1.50),
    "W": ElementProperty(74, 6, 6, 2.36),
    "Pt": ElementProperty(78, 6, 10, 2.28),
    "Au": ElementProperty(79, 6, 11, 2.54),
    "Hg": ElementProperty(80, 6, 12, 2.00),
    "Tl": ElementProperty(81, 6, 13, 1.62),
    "Pb": ElementProperty(82, 6, 14, 2.33),
    "Bi": ElementProperty(83, 6, 15, 2.02),
}

ELEMENTS = tuple(sorted(ELEMENT_PROPERTIES))
STAT_NAMES = (
    "n_elements",
    "total_atoms",
    "max_fraction",
    "composition_entropy",
    "mean_z",
    "spread_z",
    "mean_period",
    "spread_period",
    "mean_group",
    "spread_group",
    "mean_electronegativity",
    "spread_electronegativity",
    "other_fraction",
)
FEATURE_NAMES = tuple(f"frac_{element}" for element in ELEMENTS) + STAT_NAMES
TOKEN_RE = re.compile(r"([A-Z][a-z]?|\(|\)|[0-9]+(?:\.[0-9]+)?)")
HYDRATE_SEPARATOR_RE = re.compile(r"[·•]")
NUMBER_RE = re.compile(r"^[0-9]")


def parse_formula(formula: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for part in HYDRATE_SEPARATOR_RE.split(str(formula)):
        tokens = TOKEN_RE.findall(part)
        if not tokens:
            continue
        multiplier = 1.0
        if NUMBER_RE.match(tokens[0]) and len(tokens) > 1:
            multiplier = float(tokens[0])
            if multiplier <= 0:
                raise ValueError("formula amounts must be positive")
            tokens = tokens[1:]
        part_counts, pos = _parse_group(tokens, 0)
        if pos != len(tokens):
            raise ValueError(f"unparsed formula tokens in: {formula}")
        for element, amount in part_counts.items():
            counts[element] = counts.get(element, 0.0) + amount * multiplier
    if not counts:
        raise ValueError(f"could not parse formula: {formula}")
    return counts


def _parse_group(tokens: list[str], pos: int) -> tuple[dict[str, float], int]:
    counts: dict[str, float] = {}
    while pos < len(tokens):
        token = tokens[pos]
        if token == ")":
            return counts, pos + 1
        if token == "(":
            nested, pos = _parse_group(tokens, pos + 1)
            multiplier, pos = _read_multiplier(tokens, pos)
            for element, amount in nested.items():
                counts[element] = counts.get(element, 0.0) + amount * multiplier
            continue
        if not re.match(r"^[A-Z][a-z]?$", token):
            raise ValueError(f"unexpected token {token}")
        amount, pos = _read_multiplier(tokens, pos + 1)
        counts[token] = counts.get(token, 0.0) + amount
    return counts, pos


def _read_multiplier(tokens: list[str], pos: int) -> tuple[float, int]:
    if pos < len(tokens) and NUMBER_RE.match(tokens[pos]):
        value = float(tokens[pos])
        if value <= 0:
            raise ValueError("formula amounts must be positive")
        return value, pos + 1
    return 1.0, pos


def element_fractions(formula: str) -> dict[str, float]:
    counts = parse_formula(formula)
    total = sum(counts.values())
    if total <= 0:
        raise ValueError(f"formula has non-positive atom total: {formula}")
    return {element: amount / total for element, amount in counts.items()}


def weighted_mean(values: list[float], weights: list[float]) -> float:
    return sum(value * weight for value, weight in zip(values, weights))


def weighted_spread(values: list[float], weights: list[float], mean: float) -> float:
    variance = sum(
        weight * (value - mean) ** 2 for value, weight in zip(values, weights)
    )
    return math.sqrt(max(0.0, variance))


def composition_features(formula: str) -> dict[str, float]:
    counts = parse_formula(formula)
    total = sum(counts.values())
    fractions = {element: amount / total for element, amount in counts.items()}
    out = {name: 0.0 for name in FEATURE_NAMES}
    for element in ELEMENTS:
        out[f"frac_{element}"] = fractions.get(element, 0.0)
    known = [
        (element, fraction)
        for element, fraction in fractions.items()
        if element in ELEMENT_PROPERTIES
    ]
    out["other_fraction"] = sum(
        fraction
        for element, fraction in fractions.items()
        if element not in ELEMENT_PROPERTIES
    )
    out["n_elements"] = float(len(fractions))
    out["total_atoms"] = float(total)
    out["max_fraction"] = max(fractions.values())
    out["composition_entropy"] = -sum(
        frac * math.log(frac) for frac in fractions.values()
    )
    if not known:
        return out
    known_weight = sum(fraction for _, fraction in known)
    weights = [fraction / known_weight for _, fraction in known]
    for attr, mean_key, spread_key in [
        ("z", "mean_z", "spread_z"),
        ("period", "mean_period", "spread_period"),
        ("group", "mean_group", "spread_group"),
        ("electronegativity", "mean_electronegativity", "spread_electronegativity"),
    ]:
        values = [
            float(getattr(ELEMENT_PROPERTIES[element], attr)) for element, _ in known
        ]
        mean = weighted_mean(values, weights)
        out[mean_key] = mean
        out[spread_key] = weighted_spread(values, weights, mean)
    return out


def feature_vector(formula: str) -> list[float]:
    features = composition_features(formula)
    return [features[name] for name in FEATURE_NAMES]

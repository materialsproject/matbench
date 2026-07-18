"""
Reproduction script for the CompGBM Matbench bundle (matbench_glass).

Bundled source (src/) is copied byte-identical from the original workspace:
  src/run_matbench_glass.py        - official-fold runner (B0/B1/B2 models)
  src/composition_features_v2.py   - composition feature descriptor set

To reproduce from this folder:

    pip install pymatgen scikit-learn pandas numpy
    python3 src/run_matbench_glass.py

This regenerates the official-protocol B0/B1/B2 fold scores; the B2
(HistGradientBoostingClassifier) run is what results.json.gz records.
Expected headline: mean official-fold rocauc 0.8658 (matbench_glass, task
schema documented in info.json).
"""

from __future__ import annotations


def main() -> None:
    print("See src/run_matbench_glass.py for the full official-fold runner.")
    print("Run: python3 src/run_matbench_glass.py")


if __name__ == "__main__":
    main()

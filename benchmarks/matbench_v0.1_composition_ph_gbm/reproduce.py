"""
Reproduction script for the CompPHGBM Matbench bundle
(matbench_log_gvrh + matbench_log_kvrh).

Bundled source (src/) is copied byte-identical from the original workspace:
  src/run_matbench_gvrh.py         - official-fold runner for log_gvrh
  src/run_matbench_kvrh.py         - official-fold runner for log_kvrh
  src/ph_features.py               - shared persistent-homology featurizer
  src/composition_features_v2.py   - composition feature descriptor set

run_matbench_kvrh.py imports ph_features.py from the same directory, so run
both from this folder (do not move src/ contents apart).

To reproduce from this folder:

    pip install pymatgen scikit-learn pandas numpy ripser persim
    python3 src/run_matbench_gvrh.py
    python3 src/run_matbench_kvrh.py

Expected headline: mean official-fold MAE 0.0813 (log_gvrh), 0.0636
(log_kvrh); both use composition + persistent-homology structure features
with a HistGradientBoostingRegressor (task schema documented in info.json).
"""

from __future__ import annotations


def main() -> None:
    print("See src/run_matbench_gvrh.py and src/run_matbench_kvrh.py.")
    print("Run: python3 src/run_matbench_gvrh.py && python3 src/run_matbench_kvrh.py")


if __name__ == "__main__":
    main()

"""Amendment-4 configuration (FROZEN at the Amendment-4 freeze; ANALYSIS_PLAN.md Section 14).

Amendment 4 = Amendment 3 with the two changes requested in the matbench PR #372 review of 2026-11-16:
  1. NO DFT block: no candidate contains the external Materials Project block; the entry is composition-only;
  2. train-only median imputation in every split, inner CV included (src/features4.py, src/experts4.py).
Everything else is unchanged: seed 0, J = 5, 20 Optuna trials, the Amendment-2 expert pool, the Amendment-3
fold-internal selection rule (inner-CV score of the uncorrected simplex stack on T_k; ties -> simpler candidate).
"""
from .frozen_config2 import SEED, N_INNER, BLOCKS2  # noqa: F401  (unchanged: seed 0, J = 5)

N_TRIALS = 20

# The Amendment-3 block sets with the DFT block removed. F_red (= F_min + DFT) disappears.
BLOCKSETS = {
    "F_min": ("base", "extra"),                                   # Magpie, element fractions, 21 extra descriptors
    "F_all": tuple(b for b in BLOCKS2 if b != "dft"),             # the Amendment-2 set without "dft"
}
assert all("dft" not in v for v in BLOCKSETS.values())

# Candidate block sets per task, SIMPLEST FIRST (= tie-break order). The same two for all four tasks.
CAND_BLOCKS = {t: ("F_min", "F_all") for t in
               ("matbench_steels", "matbench_glass", "matbench_expt_gap", "matbench_expt_is_metal")}

# Expert-subset candidates: exactly the Amendment-3 ones (src/frozen_config3.SUBSETS).
SUBSETS = {
    "matbench_expt_gap": (
        {"hurdle": False, "drop_experts": []},
        {"hurdle": False, "drop_experts": ["et_frac"]},
        {"hurdle": True, "drop_experts": []},
        {"hurdle": True, "drop_experts": ["et_frac"]},
    ),
    "matbench_steels": ({"hurdle": False, "drop_experts": []},),
    "matbench_expt_is_metal": ({"hurdle": False, "drop_experts": []},),
    "matbench_glass": ({"hurdle": False, "drop_experts": []},),
}


def pool_cfg(task):
    return {"n_trials": N_TRIALS, "hurdle": any(s["hurdle"] for s in SUBSETS[task]), "drop_experts": []}


def final_cfg(task, subset):
    return {"n_trials": N_TRIALS, "hurdle": bool(subset["hurdle"]), "drop_experts": list(subset["drop_experts"])}


def candidate_name(blockset, subset):
    return f"{blockset}|hurdle={'on' if subset['hurdle'] else 'off'}|et_frac={'out' if 'et_frac' in subset['drop_experts'] else 'in'}"

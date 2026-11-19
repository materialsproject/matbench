"""Amendment-3 configuration (FROZEN at the Amendment-3 freeze; ANALYSIS_PLAN.md Section 13).

Amendment 3 keeps every frozen Amendment-2 setting (src/frozen_config2.py, src/experts2.py, src/pipeline2.py) and
turns the coarse design choices that Amendment 2 had fixed from the fold-0 design phase into a small CANDIDATE SET.
The choice among the candidates is made inside each official training fold T_k by the same automatic inner-CV
criterion as Amendment 1 (ANALYSIS_PLAN.md Section 8): the cross-fitted OOF score of the uncorrected global simplex
stack on T_k (MAE / ROC-AUC), J = 5, seed 0. Ties go to the earlier (simpler) candidate in the orders below.
"""
from .frozen_config2 import SEED, N_INNER, BLOCKS2  # noqa: F401  (re-exported, unchanged: seed 0, J = 5)

N_TRIALS = 20                      # Optuna budget of the tuned LightGBM expert (unchanged from Amendment 2)

# Feature-block candidate sets. "dft" is the disclosed external DFT transfer block (kept in F_red: it is disclosed
# external data, not a design choice under test). F_full is exactly the frozen Amendment-2 block set.
BLOCKSETS = {
    "F_min": ("base", "extra"),
    "F_red": ("base", "extra", "dft"),
    "F_full": BLOCKS2,
}

# Candidate block sets per task, SIMPLEST FIRST (= tie-break order). For steels and glass the DFT block is (almost)
# empty (0 / 60 matched compositions), so F_min would be the same information as F_red: only F_red vs F_full.
# For the two gap tasks F_min is a candidate only if its expert fits complete for all 5 folds within the compute
# budget (Section 13.4); if not, it is dropped for that task (recorded in fold<k>_meta.json) and said so.
CAND_BLOCKS = {
    "matbench_steels": ("F_red", "F_full"),
    "matbench_glass": ("F_red", "F_full"),
    "matbench_expt_gap": ("F_min", "F_red", "F_full"),
    "matbench_expt_is_metal": ("F_min", "F_red", "F_full"),
}

# Expert-subset candidates per task, SIMPLEST FIRST (hurdle off < on; et_frac in < out). They only change which of
# the already cross-fitted experts enter the stack, so they are scored by re-fitting the stack on the fitted
# experts' OOF predictions (no new expert fits). For the three tasks other than expt_gap the single subset is the
# frozen Amendment-2 pool.
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
    """Configuration of the FULL expert pool that is cross-fitted per (fold, block set): every expert any subset
    candidate can use. Everything else (expert settings, Optuna budget, seeds) is the frozen Amendment-2 code."""
    return {"n_trials": N_TRIALS, "hurdle": any(s["hurdle"] for s in SUBSETS[task]), "drop_experts": []}


def final_cfg(task, subset):
    """Configuration handed to the unchanged Amendment-2 pipeline (pipeline2.run_split2) once a candidate is chosen."""
    return {"n_trials": N_TRIALS, "hurdle": bool(subset["hurdle"]), "drop_experts": list(subset["drop_experts"])}


def candidate_name(blockset, subset):
    return f"{blockset}|hurdle={'on' if subset['hurdle'] else 'off'}|et_frac={'out' if 'et_frac' in subset['drop_experts'] else 'in'}"

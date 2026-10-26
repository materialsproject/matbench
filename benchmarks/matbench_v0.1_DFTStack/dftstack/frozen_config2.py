"""Amendment-2 configuration (FROZEN at the Amendment-2 freeze; ANALYSIS_PLAN.md Section 10).
All choices below were made on fold 0's TRAINING rows only (results/design_a2/, logs/design_a2*)."""
SEED = 0
N_INNER = 5
# label-free feature blocks + the disclosed DFT transfer block ("dft")
BLOCKS2 = ("base", "extra", "dft", "ep_deml", "ep_matminer", "ep_megnet", "meredig", "miedema", "yang", "ape", "wen",
           "tmetal", "cbfv_oliynyk")
_COMMON = {"n_trials": 20}
CFG2 = {
    "matbench_steels": dict(_COMMON),
    "matbench_expt_gap": dict(_COMMON, hurdle=True, drop_experts=["et_frac"]),
    "matbench_expt_is_metal": dict(_COMMON),
    "matbench_glass": dict(_COMMON),
}

from matminer.datasets import load_dataset



def load(dataset_name):


    return load_dataset(dataset_name)




LOG_KVRH = {
    "name": "log_kvrh",
    "data_file": "matbench_log_kvrh.json.gz",
    "target": "log10(K_VRH)",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None,
}

LOG_GVRH = {
    "name": "log_gvrh",
    "data_file": "matbench_log_gvrh.json.gz",
    "target": "log10(G_VRH)",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None,
}

DIELECTRIC = {
    "name": "dielectric",
    "data_file": "matbench_dielectric.json.gz",
    "target": "n",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None,
}

JDFT2D = {
    "name": "jdft2d",
    "data_file": "matbench_jdft2d.json.gz",
    "target": "exfoliation_en",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None,
}

MP_GAP = {
    "name": "mp_gap",
    "data_file": "matbench_mp_gap.json.gz",
    "target": "gap pbe",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None,
}

MP_IS_METAL = {
    "name": "mp_is_metal",
    "data_file": "matbench_mp_is_metal.json.gz",
    "target": "is_metal",
    "problem_type": AMM_CLF_NAME,
    "clf_pos_label": True,
}

MP_E_FORM = {
    "name": "mp_e_form",
    "data_file": "matbench_mp_e_form.json.gz",
    "target": "e_form",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None,
}

PEROVSKITES = {
    "name": "perovskites",
    "data_file": "matbench_perovskites.json.gz",
    "target": "e_form",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None,
}

GLASS = {
    "name": "glass",
    "data_file": "matbench_glass.json.gz",
    "target": "gfa",
    "problem_type": AMM_CLF_NAME,
    "clf_pos_label": True,
}

EXPT_IS_METAL = {
    "name": "expt_is_metal",
    "data_file": "matbench_expt_is_metal.json.gz",
    "target": "is_metal",
    "problem_type": AMM_CLF_NAME,
    "clf_pos_label": True,
}

EXPT_GAP = {
    "name": "expt_gap",
    "data_file": "matbench_expt_gap.json.gz",
    "target": "gap expt",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None,
}

PHONONS = {
    "name": "phonons",
    "data_file": "matbench_phonons.json.gz",
    "target": "last phdos peak",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None,
}

STEELS = {
    "name": "steels",
    "data_file": "matbench_steels.json.gz",
    "target": "yield strength",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None,
}
"""
This file makes the following benchmarking datasets:
    - elasticity_K_VRH
    - elasticity_log10(K_VRH)
    - elasticity_G_VRH
    - elasticity_log10(G_VRH)

From matminer's dataset library.
"""

import numpy as np
import pandas as pd
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.datasets.dataset_retrieval import load_dataset

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

mpdr = MPDataRetrieval()

df = mpdr.get_dataframe(
    criteria={
        "e_above_hull": {"$lt": 0.150},
        "formation_energy_per_atom": {"$lt": 0.150},
        "elasticity": {"$exists": 1, "$ne": None},
    },
    # "elements": },
    properties=[
        "material_id",
        "structure",
        "elasticity.K_VRH",
        "elasticity.G_VRH",
        "elasticity.G_Voigt",
        "elasticity.K_Voigt",
        "elasticity.G_Reuss",
        "elasticity.K_Reuss",
        "warnings",
    ],
    index_mpid=False,
)

df = df.rename(
    columns={
        "elasticity.K_VRH": "K_VRH",
        "elasticity.G_VRH": "G_VRH",
        "elasticity.G_Voigt": "G_Voigt",
        "elasticity.K_Voigt": "K_Voigt",
        "elasticity.G_Reuss": "G_Reuss",
        "elasticity.K_Reuss": "K_Reuss",
    }
)

df = df[
    (df["K_VRH"] > 0.0)
    & (df["G_VRH"] > 0.0)
    & (df["G_Voigt"] > 0.0)
    & (df["K_Voigt"] > 0.0)
    & (df["K_Reuss"] > 0.0)
    & (df["G_Reuss"] > 0.0)
]
df = df[
    (df["K_Reuss"] <= df["K_VRH"])
    & (df["K_VRH"] <= df["K_Voigt"])
    & (df["G_Reuss"] <= df["G_VRH"])
    & (df["G_VRH"] <= df["G_Voigt"])
]

print(df["warnings"].astype(str).value_counts())

df["log10(K_VRH)"] = np.log10(df["K_VRH"])
df["log10(G_VRH)"] = np.log10(df["G_VRH"])

df = df.reset_index(drop=True)

for target in ["K_VRH", "G_VRH", "log10(K_VRH)", "log10(G_VRH)"]:
    dftemp = df[["structure", target]]
    dftemp.to_pickle(f"elasticity_{target}.pickle.gz")

for s in df["structure"]:
    if any([e.is_noble_gas for e in s.composition.elements]):
        print(s.composition)

print(df)

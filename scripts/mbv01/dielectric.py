"""
This file makes the following benchmarking datasets:
    - castelli

Regenerating from the newest Materials Project calculations
"""

import numpy as np
import pandas as pd
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.datasets.dataset_retrieval import load_dataset
from pymatgen import Element

# pd.set_option('display.height', 1000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

mpdr = MPDataRetrieval()


# df = load_dataset("dielectric_constant")

df = mpdr.get_dataframe(
    criteria={"has": "diel"},
    properties=[
        "material_id",
        "diel.n",
        "formation_energy_per_atom",
        "e_above_hull",
        "structure",
    ],
    index_mpid=False,
)
df = df[(df["e_above_hull"] < 0.150) & (df["formation_energy_per_atom"] < 0.150)]
df = df.rename(columns={"diel.n": "n"})
df = df[(df["n"] >= 1)]
df = df.dropna()

df = df[["structure", "n"]]

# See if there is anything wrong with the Lu containing entries.
numLu = 0
for i, s in enumerate(df["structure"]):
    if Element("Lu") in s.composition.elements:
        print(s.composition.formula, df["n"].iloc[i])
        numLu += 1
print(numLu)

df = df.reset_index(drop=True)

print(df)
print(df.describe())
# df.to_pickle("dielectric.pickle.gz")

# df = pd.read_pickle("dielectric.pickle.gz")
df["is_noble"] = [
    any([e.is_noble_gas for e in s.composition.elements]) for s in df["structure"]
]
dfnoble = df[df["is_noble"]]
print("Size of noble gas containing:", dfnoble.shape)

df = df[~df["is_noble"]]
df = df.drop(columns=["is_noble"])
df = df.reset_index(drop=True)
print(df)
df.to_pickle("dielectric.pickle.gz")

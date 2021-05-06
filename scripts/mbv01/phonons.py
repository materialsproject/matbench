"""
This file makes the following benchmarking datasets:
    - phonons

From matminer's dataset library.
"""

from matminer.datasets.dataset_retrieval import load_dataset
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval


import pandas as pd

# pd.set_option('display.height', 1000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

mpdr = MPDataRetrieval()


df = load_dataset("phonon_dielectric_mp")

print(df)

mpids = df["mpid"].tolist()
dfe = mpdr.get_dataframe(
    criteria={"material_id": {"$in": mpids}},
    properties=["e_above_hull", "formation_energy_per_atom", "material_id"],
    index_mpid=False,
)
dfe = dfe.rename(columns={"material_id": "mpid"})

df = pd.merge(df, dfe, how="inner")


df = df[(df["e_above_hull"] < 0.150) & (df["formation_energy_per_atom"] < 0.150)]
df = df[["structure", "last phdos peak"]]
df = df.reset_index(drop=True)

print(df)

df.to_pickle("phonons.pickle.gz")


df = pd.read_pickle("phonons.pickle.gz")
for s in df["structure"]:
    if any([e.is_noble_gas for e in s.composition.elements]):
        print(s.composition)

print(df)

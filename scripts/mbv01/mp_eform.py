"""
This file makes the following benchmarking datasets:
    - mp_e_form

Generated from the materials project.
"""

import numpy as np
import pandas as pd
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.datasets.dataset_retrieval import load_dataset
from pymatgen import MPRester
from tqdm import tqdm

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

chunksize = 1000

mpdr = MPDataRetrieval()
mpr = MPRester()


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


df = mpdr.get_dataframe(
    criteria={"formation_energy_per_atom": {"$lt": 2.5}},
    properties=["material_id", "warnings"],
    index_mpid=False,
)

print(df["warnings"].astype(str).value_counts())

structures = pd.DataFrame(
    {"structure": [], "material_id": [], "formation_energy_per_atom": []}
)

for chunk in tqdm(chunks(range(len(df)), chunksize)):
    print(chunk[0], chunk[-1])
    mpids = df.loc[chunk[0] : chunk[-1], "material_id"].tolist()
    stchunk = mpdr.get_dataframe(
        criteria={"material_id": {"$in": mpids}},
        properties=["structure", "material_id", "formation_energy_per_atom"],
        index_mpid=False,
    )
    structures = pd.concat([stchunk, structures])


df = pd.merge(structures, df)
df = df.dropna()

# df.to_pickle("mp.pickle")


df = df.rename(columns={"formation_energy_per_atom": "e_form"})
df = df[["structure", "e_form"]]
df = df.reset_index(drop=True)
# df.to_pickle("mp_e_form.pickle.gz")
#
# print(df)

# df = pd.read_pickle("mp_e_form.pickle.gz")
# print(df.shape)
df["is_noble"] = [
    any([e.is_noble_gas for e in s.composition.elements]) for s in df["structure"]
]
dfnoble = df[df["is_noble"]]
print("Size of noble gas containing:", dfnoble.shape)

df = df[~df["is_noble"]]
df = df.drop(columns=["is_noble"])
df = df.reset_index(drop=True)
print(df)
df.to_pickle("mp_e_form.pickle.gz")

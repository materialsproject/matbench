"""
This file makes the following benchmarking datasets:
    - mp_gaps
    - mp_is_metal

From matminer's dataset library.
"""

from pymatgen import MPRester
from matminer.datasets.dataset_retrieval import load_dataset
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
import pandas as pd
import numpy as np
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
    criteria={
        "e_above_hull": {"$lt": 0.150},
        "formation_energy_per_atom": {"$lt": 0.150},
        "band_gap": {"$exists": 1, "$ne": None},
    },
    properties=["material_id", "warnings"],
    index_mpid=False,
)

print(df["warnings"].astype(str).value_counts())


structures = pd.DataFrame({"structure": [], "material_id": [], "band_gap": []})

for chunk in tqdm(chunks(range(len(df)), chunksize)):
    print(chunk[0], chunk[-1])
    mpids = df.loc[chunk[0] : chunk[-1], "material_id"].tolist()
    stchunk = mpdr.get_dataframe(
        criteria={"material_id": {"$in": mpids}},
        properties=["structure", "material_id", "band_gap"],
        index_mpid=False,
    )
    structures = pd.concat([stchunk, structures])
df = pd.merge(structures, df)
df = df.dropna()
# df.to_pickle("mp_gap_dumb.pickle")


# df = pd.read_pickle("mp_gap_dumb.pickle")

df = df.rename(columns={"band_gap": "gap pbe"})
df["is_metal"] = df["gap pbe"] == 0
df = df.reset_index(drop=True)


# df = pd.read_pickle("mp_is_metal.pickle.gz")
# print(df.shape)
df["is_noble"] = [
    any([e.is_noble_gas for e in s.composition.elements]) for s in df["structure"]
]
dfnoble = df[df["is_noble"]]
print("Size of noble gas containing:", dfnoble.shape)

df = df[~df["is_noble"]]
# df = df.drop(columns=["is_noble"])
df = df.reset_index(drop=True)
print(df)
# df.to_pickle("mp_is_metal.pickle.gz")
for target in ["gap pbe", "is_metal"]:
    dftemp = df[["structure", target]]
    dftemp.to_pickle("mp_{}.pickle.gz".format(target))

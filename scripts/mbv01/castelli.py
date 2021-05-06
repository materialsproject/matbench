"""
This file makes the following benchmarking datasets:
    - castelli

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

df = load_dataset("castelli_perovskites")
df = df[["structure", "e_form"]]
df = df.reset_index(drop=True)

print(df)
df.to_pickle("castelli.pickle.gz")

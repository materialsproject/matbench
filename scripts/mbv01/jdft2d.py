"""
This file makes the following benchmarking datasets:
    - jdft2d

From matminer's dataset library.
"""

from matminer.datasets.dataset_retrieval import load_dataset


import pandas as pd

# pd.set_option('display.height', 1000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

df = load_dataset("jarvis_dft_2d")

df = df[["structure", "exfoliation_en"]]
df = df.reset_index(drop=True)

print(df)
df.to_pickle("jdft2d.pickle.gz")

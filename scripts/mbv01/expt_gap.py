"""
This file makes the following benchmarking datasets:
    - expt_gap

From matminer's dataset library.


To combat duplicate compositions, we don't keep any compositions with a range
of bandgaps more than 0.1eV. Then, we find the mean band gap for each composition
and keep the value closest to the mean.
"""
import numpy as np
import pandas as pd
from matminer.datasets.dataset_retrieval import load_dataset
from matminer.featurizers.conversions import StrToComposition
from matminer.utils.io import store_dataframe_as_json
from tqdm import tqdm

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.set_option("precision", 8)

df = load_dataset("expt_gap")
df = df.rename(columns={"formula": "composition"})


# print("Ground Truth")
# print(df[df["composition"] == "ZrW2"])  # should be 0.00
# print(df[df["composition"] == "ZrSe2"]) # should be 2.00
# raise ValueError


excluded_compositions = []


# Prevent differences in order of formula symbols from corrupting the actual number of unique compositions
df = StrToComposition(target_col_id="composition_obj").featurize_dataframe(
    df, "composition"
)
df["composition"] = [c.reduced_formula for c in df["composition_obj"]]
df = df.drop(columns=["composition_obj"])

unique = df["composition"].unique()
print("Number of unique compositions:", len(unique))
# raise ValueError

new_df_dict = {"composition": [], "gap expt": []}
for c in tqdm(unique):
    df_per_comp_gaps = df[df["composition"] == c]
    per_comp_gaps = df_per_comp_gaps["gap expt"]
    measurement_range = max(per_comp_gaps) - min(per_comp_gaps)
    if measurement_range > 0.1:
        # print(df_per_comp_gaps)
        # big_diff += 1
        excluded_compositions.append(c)
    else:
        mean_gap = per_comp_gaps.mean()
        gap_diffs = per_comp_gaps - mean_gap
        min_gap_diff = gap_diffs.min()
        min_gap_diff_index = gap_diffs.tolist().index(min_gap_diff)
        actual_gap_diff = per_comp_gaps.tolist()[min_gap_diff_index]
        # if len(per_comp_gaps) > 1:
        #     print(f"{c} decided on {actual_gap_diff} from \n {per_comp_gaps} \n\n")
        new_df_dict["composition"].append(c)
        new_df_dict["gap expt"].append(actual_gap_diff)


df_new = pd.DataFrame(new_df_dict)
df_new = df_new.sort_values(by="composition")

df_new.to_csv("tmp.csv")


# df_new = df_new.reset_index(drop=True)
# store_dataframe_as_json(df_new, "expt_gap.json.gz", compression="gz")

print(df_new)

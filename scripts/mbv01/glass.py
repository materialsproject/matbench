"""
This file makes the following benchmarking datasets:
    - glass

This is mainly a check to make sure there are no compositions for which multiple
reports differ on whether a composition is gfa or not.

The problem compositions (those having multiple conflicting reports) are printed
out at the end. It appears there are none.
"""

import pandas as pd
from matminer.datasets.dataset_retrieval import load_dataset
from matminer.featurizers.conversions import StrToComposition
from matminer.utils.io import store_dataframe_as_json
from tqdm import tqdm

# pd.set_option('display.height', 1000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


df = load_dataset("glass_ternary_landolt")

df = df.rename(columns={"formula": "composition"})
df = df[["composition", "gfa"]]

df = StrToComposition(target_col_id="composition_obj").featurize_dataframe(
    df, "composition"
)
df["composition"] = [c.reduced_formula for c in df["composition_obj"]]
df = df.drop(columns=["composition_obj"])

# print("Ground truth")
# print(df[df["composition"]=="ZrTi9"])  # should be False in final dataframe also!!
# print(df[df["composition"]=="ZrVCo8"]) # should be True in final dataframe also!
# print(df["gfa"].value_counts())    # proportion is about 5000 GFA 2054 no GFA
# raise ValueError

unique = df["composition"].unique()
print(len(df))
print(len(unique))

problem_compositions = []
new_df_dict = {"composition": [], "gfa": []}
for c in tqdm(unique):
    df_per_comp_gfa = df[df["composition"] == c]
    per_comp_gfa = df_per_comp_gfa["gfa"]
    any_gfa = any(per_comp_gfa)
    all_gfa = any(per_comp_gfa)
    gfa = None
    if any_gfa and not all_gfa:
        print(f"Problem composition {c}: {df_per_comp_gfa}\n")
        problem_compositions.append(c)
        continue
    elif all_gfa and any_gfa:
        print(f"All gfa: {c}")
        gfa = 1
    elif not all_gfa and not any_gfa:
        print(f"No gfa: {c}")
        gfa = 0
    elif all_gfa and not any_gfa:
        raise ValueError("Impossible combination of gfa values.")

    new_df_dict["composition"].append(c)
    new_df_dict["gfa"].append(gfa)

df_new = pd.DataFrame(new_df_dict)
df_new = df_new.sort_values(by="composition")
df_new = df_new.reset_index(drop=True)

# convert to bools
df_new["gfa"] = df_new["gfa"] == 1


print(df_new)
print(df_new["gfa"].value_counts())
print(f"Problem compositions: {problem_compositions}")

store_dataframe_as_json(df_new, "glass.json.gz", compression="gz")

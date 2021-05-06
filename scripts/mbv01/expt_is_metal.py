"""
This file makes the following benchmarking datasets:
    - expt_is_metal


This is mainly a check to make sure there are no compositions for which multiple
reports differ on whether a composition is metallic or not.

The problem compositions (those having multiple conflicting reports are printed
out at the end. It appears there are none.

From matminer's dataset library.
"""
from matminer.datasets.dataset_retrieval import load_dataset
from matminer.utils.io import store_dataframe_as_json
from matminer.featurizers.conversions import StrToComposition
from tqdm import tqdm


import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

df = load_dataset("expt_gap")
df = df.rename(columns={"formula": "composition"})
print(df)
df["is_metal"] = df["gap expt"] == 0
df = df.drop(columns=["gap expt"])

# print("Ground truth")
# print(df[df["composition"]=="ZrSe3"]) # should be False in final dataframe also
# print(df[df["composition"]=="ZrW2"]) # should be True in final dataframe also
# print(df["is_metal"].value_counts())   # proportion is about 2500 metals to 4k nonmetals
# raise ValueError

df = StrToComposition(target_col_id="composition_obj").featurize_dataframe(
    df, "composition"
)
df["composition"] = [c.reduced_formula for c in df["composition_obj"]]
df = df.drop(columns=["composition_obj"])

unique = df["composition"].unique()
print("Number of unique compositions:", len(unique))

problem_compositions = []
new_df_dict = {"composition": [], "is_metal": []}
for c in tqdm(unique):
    df_per_comp_is_metal = df[df["composition"] == c]
    per_comp_is_metal = df_per_comp_is_metal["is_metal"]
    any_metals = any(per_comp_is_metal)
    all_metals = any(per_comp_is_metal)
    is_metal = None
    if not all_metals and any_metals:
        print(f"Problem composition {c}: {df_per_comp_is_metal}\n")
        problem_compositions.append(c)
        continue
    elif all_metals and any_metals:
        print(f"All metals: {c}")
        is_metal = 1
    elif not all_metals and not any_metals:
        print(f"No metals: {c}")
        is_metal = 0
    elif all_metals and not any_metals:
        raise ValueError("Impossible combination of metals.")

    new_df_dict["composition"].append(c)
    new_df_dict["is_metal"].append(is_metal)

df_new = pd.DataFrame(new_df_dict)
df_new = df_new.sort_values(by="composition")
df_new = df_new.reset_index(drop=True)

df_new["is_metal"] = df_new["is_metal"] == 1

store_dataframe_as_json(df_new, "expt_is_metal.json.gz", compression="gz")

print(df_new)
print(df_new["is_metal"].value_counts())
print(f"Problem compositions: {problem_compositions}")

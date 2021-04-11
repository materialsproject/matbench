from matbench.constants import CLF_KEY, REG_KEY
from matbench.data_ops import load

from copy import deepcopy
import pprint

from monty.serialization import loadfn


mbv01_metadata = loadfn("/Users/ardunn/alex/lbl/projects/common_env/dev_codes/matbench/matbench/matbench_v0.1_dataset_metadata.json")
new_metadata = deepcopy(mbv01_metadata)

for ds, info in mbv01_metadata.items():
    print(f"starting {ds}")
    all_data = load(ds)
    target_data = all_data[info["target"]]
    if info["task_type"] == REG_KEY:
        new_metadata[ds]["mad"] = target_data.mad()
    elif info["task_type"] == CLF_KEY:
        new_metadata[ds]["frac_true"] = target_data.mean()
    else:
        raise ValueError


pprint.pprint(new_metadata)

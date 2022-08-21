from matminer.datasets.dataset_retrieval import (
    get_all_dataset_info,
    get_available_datasets,
    load_dataset,
)

datasets = get_available_datasets(print_format=None)

for dataset in datasets:
    if "matbench_" in dataset:
        df = load_dataset(dataset)

        target_col = [col for col in df.columns if col not in ["structure", "composition"]][0]
        print(f"   * - :code:`{dataset}`\n     - :code:`{target_col}`\n     - {df.shape[0]}")


# print(get_all_dataset_info("matbench_steels"))
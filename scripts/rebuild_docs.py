import os
import json
import logging
import pprint

from monty.serialization import loadfn


from matbench.bench import MatbenchBenchmark

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DOCS_DIR = os.path.join(THIS_DIR, "../docs_src/static")
BENCHMARKS_DIR = os.path.join(THIS_DIR, "../benchmarks")


def generate_scaled_errors_plot(all_data):
    pass


def generate_general_purpose_leaderboard(all_data, benchmark_name):
    pass


def generate_per_task_leaderboards():
    pass


def generate_info_pages(all_data, benchmark_name):
    """

    """
    pass


def generate_info_page(mb: MatbenchBenchmark, info: dict, dir_name_short: str):
    is_complete = mb.is_complete

    algo_name = info["algorithm"]
    algo_desc = info["algorithm_long"]
    refs = info["bibtex_refs"]
    notes = info["notes"]

    header = f"# Data page for {algo_name} on `{mb.benchmark_name}`\n\n"
    desc = f"### Algorithm description: \n\n{algo_desc}\n\n{notes}\n\nRaw data download and example notebook available at https://github.com/hackingmaterials/matbench/tree/main/benchmarks/{dir_name_short}.\n\n"
    refs = f"### References (in bibtex format): \n\n```\n{refs}\n```\n\n"

    user_metadata = f"### User metadata:\n\n```\n{pprint.pformat(mb.user_metadata)}\n```\n\n"

    n_tasks_available = len(mb.tasks)
    n_tasks_total = len(mb.metadata.keys())

    metadata_header = f"### Metadata:\n\nTasks recorded: {n_tasks_available} of {n_tasks_total} total\n\nBenchmark is complete? {is_complete}\n\n"


    all_tasks_header = f"### Task data:\n\n"
    data_txt = ""
    for task in mb.tasks:
        task_header = f"#### `{task.dataset_name}`\n\n"

        fold_data_header = f"###### Scores and parameters on individual folds\n\n"


        fold_table = "| fold | " + " | ".join(task.scores.keys()) + " |\n" + \
                     "|------ " * (len(list(task.scores.keys())) + 1) + "|\n"
        for fold_key, fold_data in task.results.items():
            fold_line = f" | {fold_key} "

            # todo: this ordering is goofed up
            for metric_name, metric_val in fold_data.scores.items():
                formatted_val = format_float(metric_val)
                fold_line += f"| {formatted_val}"
            fold_line += " |\n"
            fold_table += fold_line
        fold_table += "\n\n"



        fold_dist_header = f"###### Score distributions among folds\n\n"
        dist_table = "| metric | mean | max | min | std |\n" \
                     "|--------|------|-----|-----|-----|\n"
        for metric_name, stats in task.scores.items():
            dist_table += f"| {metric_name} | {format_float(stats.mean)} | {format_float(stats.max)} | {format_float(stats.min)} | {format_float(stats.std)} |\n"

        dist_table += "\n\n"

        task_section = task_header + fold_data_header + fold_table + fold_dist_header + dist_table + "\n\n"
        data_txt += task_section

    final_txt = header + desc + refs + user_metadata + metadata_header + all_tasks_header + data_txt
    return final_txt


def format_float(number):
    return f"{number:.6f}"



if __name__ == "__main__":

    logging.root.setLevel(logging.DEBUG)

    all_data = {}

    # Get all benchmark data loaded into memory
    # If throws an error trying to obtain any of this data,
    # Should abort the whole docs build
    for d in os.listdir(BENCHMARKS_DIR):
        if d not in [".DS_Store", ".ipynb_checkpoints"]:
            print(d)
            d_path = os.path.join(BENCHMARKS_DIR, d)
            results_path = os.path.join(d_path, "results.json.gz")
            info_path = os.path.join(d_path, "info.json")

            # results are automatically validated, no need to validate again
            mb = MatbenchBenchmark.from_file(results_path)

            info = loadfn(info_path)

            name = info["algorithm"]
            all_data[name] = {"results": mb, "info": info}

            txt = generate_info_page(mb, info, d)
            print(txt)


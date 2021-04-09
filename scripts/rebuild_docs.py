import os
import json
import copy
import logging
import pprint
from operator import gt, lt

import tqdm
from monty.serialization import loadfn


from matbench.bench import MatbenchBenchmark
from matbench.constants import MBV01_KEY, CLF_KEY, REG_KEY
from matbench.metadata import mbv01_metadata

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(THIS_DIR, "../docs_src")
STATIC_DOCS_DIR = os.path.join(DOCS_DIR, "static")
BENCHMARKS_DIR = os.path.join(THIS_DIR, "../benchmarks")
FULL_DATA_DIR = os.path.join(DOCS_DIR, "Full Benchmark Data")


def generate_scaled_errors_plot(all_data):
    pass


def generate_general_purpose_leaderboard(all_data, benchmark_name):
    pass


def generate_per_task_leaderboards(all_data):

    all_data_per_benchmark = {}

    prefix = "Full%20Benchmark%20Data/"


    for data_packet in all_data.values():
        bmark_name = data_packet["results"].benchmark_name
        if bmark_name in all_data_per_benchmark:
            all_data_per_benchmark.append(data_packet)
        else:
            all_data_per_benchmark[bmark_name] = [data_packet]


    for bmark_name, bmarks in all_data_per_benchmark:
        if bmark_name == MBV01_KEY:
            metadata = mbv01_metadata
        else:
            raise ValueError(f"No other benchmarks configured ('{bmark_name}')")

        task_leaderboards = {t: {
            "score": None,
            "type": None
            "link": None,
            "algorithm": None
        } for t in metadata.keys()}


        gp_leaderboards = copy.deepcopy(task_leaderboards)


        for bmark_data in bmarks:
            mb = bmark_data["results"]
            info = bmark_data["info"]
            dir_name_short = bmark_data["dir_name_short"]


            for lea

            for task in mb.tasks:
                task_name = task.dataset_name

                if task.metadata.task_type == REG_KEY:
                    score = task.scores.mae.mean

                    # Better regression tasks have lower mean mae
                    op = lt
                elif task.metadata.task_type == CLF_KEY:
                    score = task.scores.rocauc.mean

                    # Better classification tasks have higher mean rocauc
                    op = gt
                else:
                    raise ValueError

                current_best_score = task_leaderboards[task_name]["score"]

                # this task's score is better or it is the first so far
                if current_best_score is None or op(score, current_best_score):
                    task_leaderboards[task_name]["score"] = score
                    task_leaderboards[task_name]["link"] = prefix + dir_name_short
                    task_leaderboards[task_name]["algorithm"] = info["algorithm"]
                    task_leaderboards[task_name]["type"] = task.metadata.task_type
                # the existing task score is best
                else:
                    pass






def generate_info_pages(all_data):
    for bmark_name, bmark_data in tqdm.tqdm(all_data.items(), desc="DOCS: FULL DATA DOCS GENERATED"):
        info = bmark_data["info"]
        mb = bmark_data["results"]
        dir_name_short = bmark_data["dir_name_short"]

        doc_str = generate_info_page(mb, info, dir_name_short)

        doc_path = os.path.join(FULL_DATA_DIR, f"{dir_name_short}.md")
        with open(doc_path, "w") as f:
            f.write(doc_str)


def generate_info_page(mb: MatbenchBenchmark, info: dict, dir_name_short: str):
    is_complete = mb.is_complete

    algo_name = info["algorithm"]
    algo_desc = info["algorithm_long"]
    refs = info["bibtex_refs"]
    notes = info["notes"]

    header = f"#`{mb.benchmark_name}`: {algo_name}\n\n"
    url = f"https://github.com/hackingmaterials/matbench/tree/main/benchmarks/{dir_name_short}"
    desc = f"### Algorithm description: \n\n{algo_desc}\n\n{notes}\n\nRaw data download and example notebook available [on the matbench repo]({url}).\n\n"
    refs = f"### References (in bibtex format): \n\n```\n{refs}\n```\n\n"

    user_metadata = f"### User metadata:\n\n```\n{pprint.pformat(mb.user_metadata)}\n```\n\n"

    n_tasks_available = len(mb.tasks)
    n_tasks_total = len(mb.metadata.keys())

    metadata_header = f"### Metadata:\n\nTasks recorded: {n_tasks_available} of {n_tasks_total} total\n\nBenchmark is complete? {is_complete}\n\n"


    all_tasks_header = f"### Task data:\n\n"
    data_txt = ""
    for task in mb.tasks:
        task_header = f"#### `{task.dataset_name}`\n\n"

        fold_data_header = f"###### Fold scores\n\n"

        # needed score order as the score order is not same between fold scores and task scores
        score_order = list(task.scores.keys())
        score_order_display = ["mape*" if s == "mape" else s for s in score_order]
        fold_table = "| fold | " + " | ".join(score_order_display) + " |\n" + \
                     "|------ " * (len(score_order) + 1) + "|\n"
        for fold_key, fold_data in task.results.items():
            fold_line = f" | {fold_key} "

            for metric_name in score_order:
                metric_val = fold_data.scores[metric_name]
                fold_line += f"| {format_float(metric_val)}"
            fold_line += " |\n"
            fold_table += fold_line
        fold_table += "\n\n"



        fold_dist_header = f"###### Fold score stats\n\n"
        dist_table = "| metric | mean | max | min | std |\n" \
                     "|--------|------|-----|-----|-----|\n"
        for metric_name, stats in task.scores.items():
            # add an asterisk next to mape since the metric is edited to not skew data on very small magnitude values
            display_name = metric_name + "*" if metric_name == "mape" else metric_name
            dist_table += f"| {display_name} | {format_float(stats.mean)} | {format_float(stats.max)} | {format_float(stats.min)} | {format_float(stats.std)} |\n"

        dist_table += "\n\n"




        params_header = "###### Fold parameters\n\n"
        params_table = "| fold | params dict|\n" \
                       "|------|------------|\n"
        for fold_key, fold_data in task.results.items():
            fold_line = f"| {fold_key} | `{fold_data.parameters}` |\n"
            params_table += fold_line

        params_table += "\n\n"

        task_section = task_header + fold_data_header + fold_table + fold_dist_header + dist_table + params_header + params_table + "\n\n"
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
            all_data[name] = {"results": mb, "info": info, "dir_name_short": d}


    print("DOCS: ALL DATA ACQUIRED")
    # generate_info_pages(all_data)
    generate_per_task_leaderboards(all_data)

import os
import json
import copy
import logging
import pprint
from operator import gt, lt

import tqdm
from monty.serialization import loadfn
import pandas as pd
import plotly.express as px

from matbench.task import MatbenchTask
from matbench.bench import MatbenchBenchmark
from matbench.constants import MBV01_KEY, CLF_KEY, REG_KEY
from matbench.metadata import mbv01_metadata

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(THIS_DIR, "../docs_src")
STATIC_DOCS_DIR = os.path.join(DOCS_DIR, "static")
BENCHMARKS_DIR = os.path.join(THIS_DIR, "../benchmarks")
FULL_DATA_DIR = os.path.join(DOCS_DIR, "Full Benchmark Data")
PER_TASK_DIR = os.path.join(DOCS_DIR, "Per-Task Leaderboards")


def generate_scaled_errors_plot(all_data):
    pass

def generate_general_purpose_leaderboard(gp_leaderboard_data_by_bmark):
    pass

def generate_per_task_leaderboards(task_leaderboard_data_by_bmark):
    for bmark_name, tasks_data in task_leaderboard_data_by_bmark.items():
        for task, entries in tasks_data.items():

            task_type = entries[0]["type"]

            if task_type == REG_KEY:

                table_data = {
                    "algorithm": [],
                    "algorithm w/ link": [],
                    "mean mae": [],
                    "std mae": [],
                    "mean rmse": [],
                    "max max_error": []
                }
            elif task_type == CLF_KEY:
                table_data = {
                    "algorithm": [],
                    "algorithm w/ link": [],
                    "mean rocauc": [],
                    "std rocauc": [],
                    "mean f1": [],
                    "mean balanced_accuracy": []
                }
            else:
                raise ValueError(f"Task type {task_type} not recognized!")

            for entry in entries:
                algo_name = entry["algorithm"]
                link = entry["link"]
                scores = entry["scores"]

                table_data["algorithm w/ link"].append(f"[{algo_name}]({link})")
                table_data["algorithm"].append(algo_name)

                if task_type == REG_KEY:
                    table_data["mean mae"].append(scores.mae.mean)
                    table_data["std mae"].append(scores.mae.std)
                    table_data["mean rmse"].append(scores.rmse.mean)
                    table_data["max max_error"].append(scores.max_error.max)
                else:
                    table_data["mean rocauc"].append(scores.rocauc.mean)
                    table_data["std rocauc"].append(scores.rocauc.std)
                    table_data["mean f1"].append(scores.f1.mean)
                    table_data["mean balanced_accuracy"].append(scores.balanced_accuracy.mean)

            df = pd.DataFrame(table_data)
            df = df.set_index("algorithm")
            sorting_column = "mean rocauc" if task_type == CLF_KEY else "mean mae"
            df = df.sort_values(by=sorting_column)

            mbt = MatbenchTask(task, autoload=False, benchmark=bmark_name)

            # header of the page
            header = f"# {bmark_name} {task}\n\n"

            subheader = f"## Individual Task Leaderboard for `{task}`\n\n"
            explanation = "_Leaderboard for an individual task. Algorithms shown here may include " \
                          "both general purpose and specialized algorithms (i.e., algorithms which " \
                          "are only valid for a subset of tasks in the benchmark._\n\n"

            info_header = "### Dataset info\n\n"

            info_body = f"##### Description\n\n{mbt.metadata.description}\n\n"
            info_body += f"Number of samples: {mbt.metadata.num_entries}\n\n"
            info_body += f"Task type: {mbt.metadata.task_type}\n\n"
            info_body += f"Input type: {mbt.metadata.input_type}\n\n"
            info_body += f"##### Dataset columns\n\n" + "".join([f"- {c}: {cd}\n" for c, cd in mbt.metadata.columns.items()]) + "\n\n"
            info_body += f"##### Dataset reference\n\n `{mbt.metadata.reference}`\n\n"

            metadata_header = "### Metadata\n\n"
            metadata = f"```\n{pprint.pformat(mbt.metadata)}\n```\n\n"

            table_header = "### Leaderboard\n\n"
            column_headers = df.columns
            table = "| " + " | ".join(column_headers) + " |\n" +  "|------" * len(column_headers) + "|\n"
            for ix, row in df.iterrows():
                table += "| "
                for th in column_headers:
                    if th == "algorithm w/ link":
                        table += f"{row[th]} | "
                    else:
                        number = format_float(row[th])
                        if th == sorting_column:
                            number = f"**{number}**"
                        table += f"{number} | "
                table += "\n"
            table += "\n"

            table = table.replace("algorithm w/ link", "algorithm")


            droppables = ["mean f1", "mean balanced_accuracy"] if task_type == CLF_KEY else ["mean rmse", "max max_error"]
            df = df.drop(columns=["algorithm w/ link"] + droppables)
            error_metric = "std rocauc" if task_type == CLF_KEY else "std mae"
            fig = px.scatter(df, error_y=error_metric, log_y=True)

            metric = "MAE" if task_type == REG_KEY else "ROCAUC"

            target = mbt.metadata.target
            lower_or_higher = "lower" if task_type == REG_KEY else "higher"
            title = f"Errors predicting '{target}' ({lower_or_higher} is better)"
            fig.update_layout(title_text=title,
                              title_font_size=15,
                              showlegend=False,
                              yaxis_title=metric,
                              xaxis_title="", paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              font={"color": "white"})
            fig.update_yaxes(linecolor="grey", gridcolor="grey")
            fig.update_xaxes(linecolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0)")

            fig_path = f"task_{bmark_name}_{task}.html"
            fig.write_html(os.path.join(STATIC_DOCS_DIR, fig_path))

            fig_reference = f'\n<iframe src="/static/{fig_path}" class="is-fullwidth" height="700px" width="1000px" frameBorder="0"> </iframe>\n\n'

            task_leaderboard_page = header + \
                                    subheader + \
                                    explanation + \
                                    table_header + \
                                    table + \
                                    fig_reference + \
                                    info_header + \
                                    info_body + \
                                    metadata_header + \
                                    metadata
            fname = os.path.join(PER_TASK_DIR, f"{bmark_name}_{task}.md")
            with open(fname, "w") as f:
                f.write(task_leaderboard_page)



def organize_task_data(all_data):
    all_data_per_benchmark = {}

    prefix = "/Full%20Benchmark%20Data/"

    for data_packet in all_data.values():
        bmark_name = data_packet["results"].benchmark_name
        if bmark_name in all_data_per_benchmark:
            all_data_per_benchmark[bmark_name].append(data_packet)
        else:
            all_data_per_benchmark[bmark_name] = [data_packet]


    gp_leaderboard_data_by_bmark = {}
    task_leaderboards_data_by_bmark = {}

    for bmark_name, bmarks in all_data_per_benchmark.items():
        if bmark_name == MBV01_KEY:
            metadata = mbv01_metadata
        else:
            raise ValueError(f"No other benchmarks configured ('{bmark_name}')")

        gp_leaderboard = {t: {
            "score": None,
            "type": None,
            "link": None,
            "algorithm": None
        } for t in metadata.keys()}

        task_leaderboards = {t: [] for t in metadata.keys()}

        for bmark_data in bmarks:
            mb = bmark_data["results"]
            info = bmark_data["info"]
            dir_name_short = bmark_data["dir_name_short"]

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


                # Include both GP and structure-required algos on
                # GP leaderboard, as there are 9 structure problems
                # across multiple dataset sizes
                if mb.is_complete or mb.is_structure_complete:
                    current_best_score = gp_leaderboard[task_name]["score"]

                    # this task's score is better or it is the first so far
                    if current_best_score is None or op(score, current_best_score):
                        gp_leaderboard[task_name]["score"] = score
                        gp_leaderboard[task_name]["link"] = prefix + dir_name_short
                        gp_leaderboard[task_name]["algorithm"] = info["algorithm"]
                        gp_leaderboard[task_name]["type"] = task.metadata.task_type
                    # the existing task score is best
                    else:
                        pass

                # Add it to the task-specific leaderboard, as all entries will be included
                # there
                task_leaderboards[task_name].append({
                    "scores": task.scores,
                    "link": prefix + dir_name_short,
                    "algorithm": info["algorithm"],
                    "type": task.metadata.task_type
                })

            gp_leaderboard_data_by_bmark[bmark_name] = gp_leaderboard
            task_leaderboards_data_by_bmark[bmark_name] = task_leaderboards

        return gp_leaderboard_data_by_bmark, task_leaderboards_data_by_bmark


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

    gp_leaderboard_data_by_bmark, task_leaderboards_data_by_bmark = organize_task_data(all_data)

    generate_per_task_leaderboards(task_leaderboards_data_by_bmark)

    # pprint.pprint(gp_leaderboard_data_by_bmark)
    # pprint.pprint(task_leaderboards_data_by_bmark)
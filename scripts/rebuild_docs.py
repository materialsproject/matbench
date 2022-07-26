import json
import os
import glob
import logging
import pprint
from operator import gt, lt

import tqdm
from monty.serialization import loadfn
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np

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
PER_TASK_DIR = os.path.join(DOCS_DIR, "Leaderboards Per-Task")
PER_TASK_DIR_PREFIX = "Leaderboards%20Per-Task/"
FULL_DATA_DIR_PREFIX = "Full%20Benchmark%20Data/"
METADATA_DIR = os.path.join(DOCS_DIR, "Benchmark Info")
METADATA_DIR_PREFIX = "Benchmark%20Info/"
SNIPPETS_DIR = os.path.join(THIS_DIR, "doc_snippets")
SCALED_ERRORS_FILENAME = "scaled_errors_{bmark_name}.html"


MP_WEBSITE_STATICS = os.path.join(STATIC_DOCS_DIR, "mp_srcs")
SCALED_ERRORS_PATH = os.path.join(STATIC_DOCS_DIR, SCALED_ERRORS_FILENAME)
SCALED_ERRORS_JSON_PATH = SCALED_ERRORS_PATH.replace(".html", ".json")


def generate_scaled_errors_graph(gp_graph_data_by_bmark):
    """
    Generate the scaled errors graph


    Args:
        gp_graph_data_by_bmark:

    Returns:

    """
    for bmark_name, tasks_data in gp_graph_data_by_bmark.items():

            if bmark_name == MBV01_KEY:

                symbols = {
                    "matbench_steels": "σᵧ",
                    "matbench_jdft2d": "Eˣ",
                    "matbench_phonons": "ωᵐᵃˣ",
                    "matbench_dielectric": "𝑛",
                    "matbench_expt_gap": "Eᵍ",
                    "matbench_expt_is_metal": "Expt. Metallicity",
                    "matbench_glass": "Metallic Glass",
                    "matbench_log_kvrh": "log₁₀Kᵛʳʰ",
                    "matbench_log_gvrh": "log₁₀Gᵛʳʰ",
                    "matbench_perovskites": "Eᶠ",
                    "matbench_mp_gap": "Eᵍ",
                    "matbench_mp_is_metal": "Metallicity",
                    "matbench_mp_e_form": "Eᶠ"
                }

                descriptors = {
                    "matbench_steels": "Steel alloys",
                    "matbench_jdft2d": "2D Materials",
                    "matbench_phonons": "Phonons",
                    "matbench_dielectric": "",
                    "matbench_expt_gap": "Experimental",
                    "matbench_expt_is_metal": "Classification",
                    "matbench_glass": "Classification",
                    "matbench_log_gvrh": "",
                    "matbench_log_kvrh": "",
                    "matbench_perovskites": "Perovskites, DFT",
                    "matbench_mp_gap": "DFT",
                    "matbench_mp_is_metal": "DFT",
                    "matbench_mp_e_form": "DFT"
                }

                metadata = mbv01_metadata
            else:
                raise ValueError(
                    f"Only {MBV01_KEY} defined as valid benchmark! '{bmark_name}' not supported.")

            # should take care of missing entries (i.e., structure only) automatically
            df = pd.DataFrame(tasks_data)

            # make scaled data for heatmap coloring
            # scale regression problems by mad/mae
            def scale_regression_problem(series, mad):
                mask = series > 0.0
                mask_iix = np.where(mask)
                series.iloc[mask_iix] = series.iloc[mask_iix] / mad
                series.loc[~mask] = np.nan
                return series

            def scale_classification_problem(series, mad):
                mask = series > 0.0
                mask_iix = np.where(mask)
                series.iloc[mask_iix] = 1 - (series.iloc[mask_iix] - 0.5) / 0.5
                # series.iloc[mask_iix] = 1 - series.iloc[mask_iix]
                series.loc[~mask] = np.nan
                return series

            scaled_df = df.copy(deep=True)
            for task in scaled_df.columns:
                if metadata[task].task_type == CLF_KEY:
                    scaler = scale_classification_problem
                else:
                    scaler = scale_regression_problem

                scaled_df[task] = scaler(scaled_df[task], metadata[task].mad)

            scaled_df = scaled_df.T
            scaled_df["n_samples"] = [metadata[task].n_samples for task in scaled_df.index]
            scaled_df["Problem"] = [f"{symbols[task]} {descriptors[task]}" for task in scaled_df.index]
            scaled_df = scaled_df.sort_values(by="n_samples")
            scaled_df.index = scaled_df["Problem"]
            scaled_df = scaled_df.drop(columns=["n_samples", "Problem"])

            best_values = scaled_df.min(axis=1)
            best_algos = scaled_df.idxmin(axis=1)

            fig = px.scatter(scaled_df, log_y=True)
            fig.update_traces(
                marker={'size': 10},
                hovertemplate="<br>".join(
                    [
                        # "Algorithm: %{text}",
                        "Problem: %{x}",
                        "Scaled Error: %{y}"
                    ]
                )
            )

            fig.update_layout(title_text="Scaled Errors",
                              title_font_size=30,
                              legend_font_size=15,
                              legend_title_font_size=15,
                              legend_title_text="Algorithm",
                              yaxis_title="Scaled MAE (regression) or <br> (1-ROCAUC)/0.5 (classification)",
                              xaxis_title="", paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              font={"color": "white"})

            # add scatter for the best algorithms on scaled error
            fig.add_trace(
                go.Scatter(
                    mode="markers",
                    x=best_values.index,
                    y=best_values,
                    marker=dict(
                        color="yellow",
                        size=10,
                    ),
                    hovertemplate= \
                        'Algorithm: %{text}<br>Problem: %{x}<br>Scaled Error: %{y}<br>',
                    text=best_algos,
                    visible="legendonly",
                    name="Best algorithms"
                )
            )
            fig.update_xaxes(linecolor="grey", gridcolor="grey")
            fig.update_yaxes(linecolor="grey", gridcolor="grey")
            fig.write_html(SCALED_ERRORS_PATH.format(bmark_name=bmark_name))


            # Update layout for showing on white background on mp website
            fig.update_layout(
                title_text="",
                font={"color": "black"}
            )
            fig.write_json(SCALED_ERRORS_JSON_PATH.format(bmark_name=bmark_name))


# NOTE: MUST BE CALLED AFTER CREATING generate_scaled_errors_graph
def generate_general_purpose_leaderboard_and_plot(gp_leaderboard_data_by_bmark):
    """
    Generate both the general purpose leaderboard and scaled errors graph on the main leaderboard page.


    THE SCALED ERRORS GRAPH MUST EXIST FOR THIS FUNCTION TO COMPLETE CORRECTLY.
    Call generate_scaled_errors_graph to create this graph.

    This function takes in the doc snippet for index.md and prepends the leaderboard to it.

    So to make changes to the index.md you should change the doc snippet, not the index.md file in docs_src.


    Args:
        gp_leaderboard_data_by_bmark:

    Returns:

    """
    gp_leaderboard_txt = ""
    for bmark, gp_data in gp_leaderboard_data_by_bmark.items():

        if bmark == MBV01_KEY:
            metadata = mbv01_metadata
        else:
            raise ValueError(f"{bmark} not a valid benchmark!")

        table_data = {
            "task": [k for k in gp_data.keys()],
            "n_samples": [metadata[k].n_samples for k in gp_data.keys()],
            "algorithm": [d["algorithm"] for d in gp_data.values()],
            "completeness": [d["completeness"] for d in gp_data.values()],
            "link": [d["link"] for d in gp_data.values()],
            "score": [d["score"] for d in gp_data.values()],
            "type": [d["type"] for d in gp_data.values()]
        }

        df_src = pd.DataFrame(table_data).sort_values(by="n_samples")
        table_header = f"## Leaderboard: General Purpose Algorithms on `{bmark}`\n\n"
        table_explanation = f"Find more information about this benchmark on [the benchmark info page]({METADATA_DIR_PREFIX}{bmark}.md)\n\n"
        table = "| Task name | Samples | Algorithm | Verified MAE (unit) or ROCAUC | Notes |\n" \
                "|------------------|---------|-----------|----------------------|-------|\n"
        # create leaderboard table
        for _, row in df_src.iterrows():

            task_name = f"{row['task']}"
            task_name_link = f"[{task_name}]({PER_TASK_DIR_PREFIX}matbench_v0.1_{task_name}.md)"
            samples = format_int(row["n_samples"])
            algorithm = f"[{row['algorithm']}]({row['link']}.md)"

            task_metadata = metadata[row['task']]

            score = f"{format_float(row['score'])} ({task_metadata.unit})" if task_metadata.task_type == REG_KEY else f"{format_float(row['score'])}"
            score = f"**{score}**"

            if row["completeness"] == "structure":
                notes = "structure required"
            elif row["completeness"] == "all":
                notes = ""
            else:
                raise ValueError(f"{row['completeness']} is not a valid type of general purpose completeness!")

            table += f"| {task_name_link} | {samples} | {algorithm} | {score} | {notes} |\n"
        table += "\n\n"
        scaled_errors_plot_txt = f'\n<iframe src="static/{SCALED_ERRORS_FILENAME.format(bmark_name=bmark)}" class="is-fullwidth" height="700px" width="1000px" frameBorder="0"> </iframe>\n\n'
        gp_leaderboard_txt += table_header + table_explanation + table + scaled_errors_plot_txt

    # Load the static index from the snippets dir
    with open(os.path.join(SNIPPETS_DIR, "index.md"), encoding="utf-8") as f:
        static_txt = f.read()


    page_header = f"# Leaderboard\n\n"
    final_txt = page_header + gp_leaderboard_txt + static_txt

    with open(os.path.join(DOCS_DIR, "index.md"), "w", encoding="utf-8") as f:
        print("Writing leaderboard and plot to index.md...")
        f.write(final_txt)

    with open(os.path.join(STATIC_DOCS_DIR,  "gp_table.json"), "w", encoding="utf-8") as f:
        print("Writing static gp_table json to gp_table.json...")
        j = {"txt": gp_leaderboard_txt}
        json.dump(j, f)


def generate_per_task_leaderboards(task_leaderboard_data_by_bmark):
    """
    Generate all .md files for the per-task leaderboards.


    Args:
        task_leaderboard_data_by_bmark:

    Returns:

    """
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
                # link must be relative, and since this is inside a dir, has to reference outside
                link = "../" + entry["link"]
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
            sorting_order = True if task_type == REG_KEY else False
            df = df.sort_values(by=sorting_column, ascending=sorting_order)

            mbt = MatbenchTask(task, autoload=False, benchmark=bmark_name)

            # header of the page
            header = f"# {bmark_name} {task}\n\n"

            subheader = f"## Individual Task Leaderboard for `{task}`\n\n"
            explanation = "_Leaderboard for an individual task. Algorithms shown here may include " \
                          "both general purpose and specialized algorithms (i.e., algorithms which " \
                          "are only valid for a subset of tasks in the benchmark._\n\n"

            info_header = "### Dataset info\n\n"

            info_body = f"##### Description\n\n{mbt.metadata.description}\n\n"
            info_body += f"Number of samples: {mbt.metadata.n_samples}\n\n"
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

            metric = f"MAE {mbt.metadata.unit}" if task_type == REG_KEY else "ROCAUC"

            target = mbt.metadata.target
            lower_or_higher = "lower" if task_type == REG_KEY else "higher"
            title_prefix = "Errors predicting" if task_type == REG_KEY else \
                "Classification ROCAUC predicting"
            title = f"{title_prefix} '{target}' ({lower_or_higher} is better)"
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

            fig_reference = f'\n<iframe src="../../static/{fig_path}" class="is-fullwidth" height="700px" width="1000px" frameBorder="0"> </iframe>\n\n'

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
            with open(fname, "w", encoding="utf-8") as f:
                print(f"Creating task leaderboard page {fname}")
                f.write(task_leaderboard_page)


def organize_task_data(all_data):
    """
    Preprocessing step for organizing all benchmark data in specific formats,
    to be handed downstream to .md-generating functions.


    Args:
        all_data:

    Returns:

    """
    all_data_per_benchmark = {}

    prefix = FULL_DATA_DIR_PREFIX

    for data_packet in all_data.values():
        bmark_name = data_packet["results"].benchmark_name
        if bmark_name in all_data_per_benchmark:
            all_data_per_benchmark[bmark_name].append(data_packet)
        else:
            all_data_per_benchmark[bmark_name] = [data_packet]


    gp_leaderboard_data_by_bmark = {}
    task_leaderboards_data_by_bmark = {}
    gp_graph_data_by_bmark = {}

    for bmark_name, bmarks in all_data_per_benchmark.items():
        if bmark_name == MBV01_KEY:
            metadata = mbv01_metadata
        else:
            raise ValueError(f"No other benchmarks configured ('{bmark_name}')")

        gp_leaderboard = {t: {
            "score": None,
            "type": None,
            "link": None,
            "algorithm": None,
            "completeness": None
        } for t in metadata.keys()}

        task_leaderboards = {t: [] for t in metadata.keys()}

        gp_graph_data = {t: {} for t in metadata.keys()}

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


                # Include both GP, structure-required, and strcuture/comp regression algos on
                # GP leaderboard, as there are 9 structure problems or 10 regresison problems
                # across multiple dataset sizes (composition only or clf only are not included)
                if mb.is_complete or mb.is_structure_complete or mb.is_regression_complete:
                    current_best_score = gp_leaderboard[task_name]["score"]

                    gp_graph_data[task_name][info["algorithm"]] = score

                    # this task's score is better or it is the first so far
                    if current_best_score is None or op(score, current_best_score):
                        gp_leaderboard[task_name]["score"] = score
                        gp_leaderboard[task_name]["link"] = prefix + dir_name_short
                        gp_leaderboard[task_name]["algorithm"] = info["algorithm"]
                        gp_leaderboard[task_name]["type"] = task.metadata.task_type

                        if mb.is_complete or mb.is_regression_complete:
                            gp_leaderboard[task_name]["completeness"] = "all"
                        else:
                            gp_leaderboard[task_name]["completeness"] = "structure"
                    # the existing task score is best
                    else:
                        pass

                # Add it to the task-specific leaderboard, as all entries will be included
                # there
                task_leaderboards[task_name].append({
                    "scores": task.scores,
                    "link": prefix + dir_name_short + ".md",
                    "algorithm": info["algorithm"],
                    "type": task.metadata.task_type
                })

            gp_leaderboard_data_by_bmark[bmark_name] = gp_leaderboard
            task_leaderboards_data_by_bmark[bmark_name] = task_leaderboards
            gp_graph_data_by_bmark[bmark_name] = gp_graph_data

    return gp_leaderboard_data_by_bmark, task_leaderboards_data_by_bmark, gp_graph_data_by_bmark


def generate_metadata_pages(task_leaderboard_data_by_bmark):
    """
    Generate all benchmark info pages. E.g., the table for matbench_v0.1 with all the download links.

    There should be one per benchmark version - e.g., one for matbench_v0.1, one for matbench_v0.2, ...


    Args:
        task_leaderboard_data_by_bmark:

    Returns:

    """
    for bmark_name, bmark_data in task_leaderboard_data_by_bmark.items():
        metadata = MatbenchBenchmark(benchmark=bmark_name, autoload=False).metadata

        d = {}
        for task, infod in metadata.items():
            d[task] = {
                "Task name": f"`{task}`",
                "Task type": infod.task_type,
                "Target column (unit)": f"`{infod.target}` " + f"({infod.unit})" if infod.unit else f"`{infod.target}`",
                "Input type": infod.input_type,
                "Samples": infod.n_samples,
                "MAD (regression) or Fraction True (classification)": format_float(infod.mad if infod.task_type == REG_KEY else infod.frac_true),
                "Links": f"[download](https://ml.materialsproject.org/projects/{task}.json.gz), [interactive](https://ml.materialsproject.org/projects/{task})",
                "Submissions": f"{len(bmark_data[task])}"
            }

        df = pd.DataFrame(d).T.sort_values(by="Samples")

        df["Samples"] = [format_int(i) for i in df["Samples"]]
        table_header = f"# Benchmark info for `{bmark_name}`\n\n"
        table_explanation = f"The `{bmark_name}` benchmark contains {len(metadata)} tasks:\n\n"
        table = "| " + " | ".join(df.columns) + "|\n" + \
            "|-------" * len(df.columns) + "|\n"
        for _, row in df.iterrows():
            table_line = "|"
            for c in df.columns:
                table_line += f" {row[c]} |"

            table_line += "\n"
            table += table_line

        page = table_header + table_explanation + table

        path = os.path.join(METADATA_DIR, f"{bmark_name}.md")
        with open(path, "w", encoding="utf-8") as f:
            print(f"Writing benchmark info page {path}")
            f.write(page)


def generate_info_pages(all_data):
    """
    Generate all full data benchmarks pages. There should be one per submitted benchmark.

    E.g., one for /benchmarks/matbench_v0.1_algo1, one for /benchmarks/matbench_v0.1_algo2, etc.

    Args:
        all_data:

    Returns:

    """
    for bmark_name, bmark_data in tqdm.tqdm(all_data.items(), desc="DOCS: FULL DATA DOCS GENERATED"):
        info = bmark_data["info"]
        mb = bmark_data["results"]
        dir_name_short = bmark_data["dir_name_short"]

        doc_str = generate_info_page(mb, info, dir_name_short)

        doc_path = os.path.join(FULL_DATA_DIR, f"{dir_name_short}.md")
        with open(doc_path, "w", encoding="utf-8") as f:
            print(f"Writing full benchmark data page {doc_path}")
            f.write(doc_str)


def generate_info_page(mb: MatbenchBenchmark, info: dict, dir_name_short: str):
    """
    Generate a single full data benchmark page.


    Args:
        mb:
        info:
        dir_name_short:

    Returns:

    """
    is_complete = convert_bool_to_unicode_check(mb.is_complete)
    structure_complete = convert_bool_to_unicode_check(mb.is_structure_complete)
    composition_complete = convert_bool_to_unicode_check(mb.is_composition_complete)
    regression_complete = convert_bool_to_unicode_check(mb.is_regression_complete)
    classification_complete = convert_bool_to_unicode_check(mb.is_classification_complete)

    algo_name = info["algorithm"]
    algo_desc = info["algorithm_long"]
    refs = info["bibtex_refs"]
    notes = info["notes"]
    requirements = info["requirements"]

    header = f"# {mb.benchmark_name}: {algo_name}\n\n"
    url = f"https://github.com/hackingmaterials/matbench/tree/main/benchmarks/{dir_name_short}"
    desc = f"### Algorithm description: \n\n{algo_desc}\n\n#### Notes:\n{notes}\n\nRaw data download and example notebook available [on the matbench repo]({url}).\n\n"
    refs = f"### References (in bibtex format): \n\n```\n{pprint.pformat(refs)}\n```\n\n"

    user_metadata = f"### User metadata:\n\n```\n{pprint.pformat(mb.user_metadata)}\n```\n\n"

    n_tasks_available = len(mb.tasks)
    n_tasks_total = len(mb.metadata.keys())

    metadata_header = f"### Metadata:\n\n"

    metadata_table = f"| tasks recorded | {n_tasks_available}/{n_tasks_total} |\n" \
                     f"|----------------|-------------------------------------|\n" \
                     f"| complete? | {is_complete} | \n" \
                     f"| composition complete? | {composition_complete} | \n" \
                     f"| structure complete? | {structure_complete} | \n" \
                     f"| regression complete? | {regression_complete} | \n" \
                     f"| classification complete? | {classification_complete} | \n\n"

    metadata_header += metadata_table

    requirements_header = f"### Software Requirements\n\n"
    requirements_body = f"```\n{pprint.pformat(requirements)}\n```\n\n"

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
            params_short = f"{fold_data.parameters}"[:200]
            params_short = params_short + "..." if len(params_short) == 200 else params_short
            fold_line = f"| {fold_key} | `{params_short}` |\n"
            params_table += fold_line

        params_table += "\n\n"

        task_section = task_header + fold_data_header + fold_table + fold_dist_header + dist_table + params_header + params_table + "\n\n"
        data_txt += task_section

    final_txt = header + desc + refs + user_metadata + metadata_header + requirements_header + requirements_body + all_tasks_header + data_txt
    return final_txt


def format_float(number):
    return f"{number:.4f}"


def format_int(number):
    return f'{number:,}'


def nuke_docs(check=True):
    """
    Clean the source data directory.

    Args:
        check (bool): don't actually delete anything, just output the files to-be-deleted.

    Returns:

    """
    count = 0

    index = os.path.join(DOCS_DIR, "index.md")
    if os.path.exists(index):
        if not check:
            os.remove(index)
        print(f"\tdeleting index md file'{index}'")
    count += 1

    html_statics = glob.glob(os.path.join(STATIC_DOCS_DIR, "*.html"))
    for hs in html_statics:
        if not check:
            os.remove(hs)
        print(f"\tdeleting static html '{hs}'")
    count += len(html_statics)

    json_statics = glob.glob(os.path.join(STATIC_DOCS_DIR, "*.json"))
    for js in json_statics:
        if not check:
            os.remove(js)
        print(f"\tdeleting static json '{js}'")
    count += len(json_statics)

    per_task_leaderboards = glob.glob(os.path.join(PER_TASK_DIR, "*.md"))
    for pl in per_task_leaderboards:
        if not check:
            os.remove(pl)
        print(f"\tdeleting per task leaderboard md file '{pl}'")
    count += len(per_task_leaderboards)

    full_benchmark_mds = glob.glob(os.path.join(FULL_DATA_DIR, "*.md"))
    for fb in full_benchmark_mds:
        if not check:
            os.remove(fb)
        print(f"\tdeleting full benchmark md file '{fb}'")
    count += len(full_benchmark_mds)

    benchmark_info_mds = glob.glob(os.path.join(METADATA_DIR, "*.md"))
    for bi in benchmark_info_mds:
        if "notes.md" not in bi:
            if not check:
                os.remove(bi)
            print(f"\tdeleting benchmark info page '{bi}'")
            count += 1

    print(f"\tdeleted {count} files from {DOCS_DIR}")


def convert_bool_to_unicode_check(t_or_f):
    return "✓" if t_or_f else "✗"


if __name__ == "__main__":

    logging.root.setLevel(logging.DEBUG)

    # nuke_docs(check=True)
    nuke_docs(check=False)

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


    gp_leaderboard_data_by_bmark, task_leaderboards_data_by_bmark, gp_graph_data_by_bmark = organize_task_data(all_data)


    print("DOCS: ALL DATA ACQUIRED")

    generate_info_pages(all_data)

    generate_per_task_leaderboards(task_leaderboards_data_by_bmark)

    generate_metadata_pages(task_leaderboards_data_by_bmark)

    # must be called before generating the gp leaderboard
    generate_scaled_errors_graph(gp_graph_data_by_bmark)

    generate_general_purpose_leaderboard_and_plot(gp_leaderboard_data_by_bmark)



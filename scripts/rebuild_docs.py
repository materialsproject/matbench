import copy
import glob
import html
import json
import logging
import os
import pprint
from operator import gt, lt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import tqdm
from monty.serialization import loadfn

from matbench.bench import MatbenchBenchmark
from matbench.constants import CLF_KEY, MBV01_KEY, REG_KEY
from matbench.metadata import mbv01_metadata
from matbench.leaderboard import classification_score_report, prepare_classification_scores
from matbench.task import MatbenchTask

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

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

# Shared policy for current classification rankings; historical artifacts are preserved.
CLF_ROCAUC_NOTE = (
    '!!! info "Probability-based ROC AUC"\n'
    "    Classification rankings use ROC AUC recomputed from stored floating-point "
    "probabilities and the official test labels, separately for each of the five folds. "
    "The displayed mean and standard deviation are across folds. Submissions containing "
    "boolean predictions are excluded from these rankings. Historical balanced accuracy "
    "results remain available in the collapsed section on each classification task page. "
    "See [#137](https://github.com/materialsproject/matbench/issues/137) and "
    "[#181](https://github.com/materialsproject/matbench/issues/181).\n\n"
)

# Pointer to Matbench Discovery, which has its own leaderboard; see issue #388
MATBENCH_DISCOVERY_NOTE = "!!! tip \"Benchmarking ML for materials discovery?\"\n" \
                          "    For DFT-based materials discovery campaigns, use " \
                          "[Matbench Discovery](https://matbench-discovery.materialsproject.org/).\n\n"


# mp website statics locations
SCALED_ERRORS_PATH = os.path.join(STATIC_DOCS_DIR, "scaled_errors_{bmark_name}.json")
SCALED_ERRORS_NON_GP_PATH = os.path.join(STATIC_DOCS_DIR, "scaled_errors_non_gp_{bmark_name}.json")

pio.templates.default = "plotly_white"


def generate_scaled_errors_graph(gp_graph_data_by_bmark, output_fname=SCALED_ERRORS_PATH, title_txt="General purpose", title_subtext=""):
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
                # AUC=0 and AUC=1 are valid scores, not missing-data sentinels.
                mask = np.isfinite(series) & series.ge(0.0) & series.le(1.0)
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

            available = scaled_df.dropna(axis=0, how="all").dropna(axis=1, how="all").sort_index(axis=1)
            if available.empty:
                return "<p>No eligible scores are available for this chart.</p>"
            best_values = available.min(axis=1)
            best_algos = available.idxmin(axis=1)

            # Perfect classification has zero scaled error. Use a linear axis
            # when needed so it remains visible without substituting an epsilon.
            use_log_scale = not available.eq(0.0).any().any()
            fig = px.scatter(available, log_y=use_log_scale)
            if not use_log_scale:
                fig.update_yaxes(type="linear", rangemode="tozero")
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

            scale_note = "<br><sup>Linear scale includes zero scaled error.</sup>" if not use_log_scale else ""
            fig.update_layout(title_text=f"{title_txt} Scaled Errors {title_subtext}{scale_note}",
                              title_font_size=24,
                              legend_font_size=14,
                              legend_title_font_size=14,
                              legend_title_text="Algorithm",
                              yaxis_title="Scaled MAE (regression) or <br> (1-ROCAUC)/0.5 (classification)",
                              xaxis_title="",
                              paper_bgcolor="rgba(0, 0, 0, 0)",
                              plot_bgcolor="rgba(0, 0, 0, 0)",
                              font={"color": "white"},
                              legend={"yanchor": "top", "y": -0.10, "xanchor": "left", "x": 0.01})

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
            fig.update_traces(hoverlabel={"bgcolor": "black"}, selector=dict(type='scatter'))
            fig.update_xaxes(linecolor="grey", gridcolor="grey")
            fig.update_yaxes(linecolor="grey", gridcolor="grey")
            fig_div = pio.to_html(fig, full_html=False, default_height="1200px")


            if output_fname in (SCALED_ERRORS_PATH, SCALED_ERRORS_NON_GP_PATH):
                # Update layout and write files for showing on white background on mp website
                output_path = output_fname.format(bmark_name=bmark_name)
                fig.update_layout(
                    title_text="",
                    font={"color": "black"}
                )
                print(f"Writing scaled errors graph to {output_path}")
                fig.write_json(output_path)
                fig.write_html(output_path.replace(".json", ".html"))
            else:
                print(f"Output filename {output_fname} not required to write to disk.")

            return fig_div


# NOTE: MUST BE CALLED AFTER CREATING generate_scaled_errors_graph
def generate_general_purpose_leaderboard_and_plot(gp_leaderboard_data_by_bmark, total_info_counts, leaderboard_divs=None):
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
    if not leaderboard_divs:
        print("No leaderboard divs generated! No GP/NonGP leaderboards will be included in the docs...")
        leaderboard_divs = []

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
        table_header = f"## Leaderboard-Property: General Purpose Algorithms on `{bmark}`\n\n"
        table_explanation = f"Find more information about this benchmark on [the benchmark info page]({METADATA_DIR_PREFIX}{bmark}.md)\n\n"
        table = "| Task name | Samples | Algorithm | Verified MAE (unit) or ROCAUC | Notes |\n" \
                "|------------------|---------|-----------|----------------------|-------|\n"
        # create leaderboard table
        for _, row in df_src.iterrows():

            task_name = f"{row['task']}"
            task_name_link = f"[{task_name}]({PER_TASK_DIR_PREFIX}matbench_v0.1_{task_name}.md)"
            samples = format_int(row["n_samples"])
            if row["algorithm"] is None:
                missing = "No eligible general-purpose probability submissions" if metadata[row["task"]].task_type == CLF_KEY else "No general-purpose submissions"
                table += f"| {task_name_link} | {samples} | — | — | {missing} |\n"
                continue
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
        # scaled_errors_plot_txt = f'\n<iframe src="static/{SCALED_ERRORS_FILENAME.format(bmark_name=bmark)}" class="is-fullwidth" height="1200px" width="1000px" frameBorder="0" allowtransparency="true"> </iframe>\n\n'
        # scaled_errors_non_gp_plt_txt = f'\n<iframe src="static/{SCALED_ERRORS_NON_GP_FILENAME.format(bmark_name=bmark)}" class="is-fullwidth" height="1200px" width="1000px" frameBorder="0"> </iframe>\n\n'
        audit_link = "[Download classification score audit (JSON)](./static/classification_scores.json)\n\n"
        gp_leaderboard_txt += table_header + table_explanation + CLF_ROCAUC_NOTE + audit_link + table

        for div_txt in leaderboard_divs:
            gp_leaderboard_txt += div_txt

    # Load the static index from the snippets dir
    with open(os.path.join(SNIPPETS_DIR, "index_0.md"), encoding="utf-8") as f:
        static_txt_0 = f.read()

    with open(os.path.join(SNIPPETS_DIR, "index.md"), encoding="utf-8") as f:
        static_txt = f.read()

    n_algos = total_info_counts["algos"]
    n_tasks = total_info_counts["tasks"]
    n_benchmarks = total_info_counts["benchmarks"]

    page_header = f"# Leaderboard \n\n" \
                  f"**Matbench is an automated leaderboard** for benchmarking state of the art ML algorithms predicting a **diverse range of solid materials' properties**. " \
                  f"It is hosted and maintained by [the Materials Project](https://materialsproject.org). \n\n ![crystal](static/crystals.png)\n\n" \
                  f"{MATBENCH_DISCOVERY_NOTE}"

    # Define the table content as a nested list
    table_header = "|             | [<span style='color: #1e90ff'>Materials Properties</span>](/#leaderboard-property-general-purpose-algorithms-on-matbench_v01) |\n" \
               "|-------------|---------------------|\n"

    table_rows = f"| Task Submissions |        `{n_tasks}`          |\n" \
                f"| Algorithms      |          `{n_algos}`         |\n" \
                f"| Benchmark Task Suite |         `{n_benchmarks}`        |\n\n"


    load_plotly = f"<script src='https://cdn.plot.ly/plotly-latest.min.js'></script> \n\n"
    final_txt = load_plotly + page_header + table_header + table_rows + gp_leaderboard_txt + static_txt_0 + "\n" + static_txt


    with open(os.path.join(DOCS_DIR, "index.md"), "w", encoding="utf-8") as f:
        print("Writing leaderboard and plot to index.md...")
        f.write(final_txt)

    with open(os.path.join(STATIC_DOCS_DIR,  "gp_table.json"), "w", encoding="utf-8") as f:
        print("Writing static gp_table json to gp_table.json...")
        j = {"txt": gp_leaderboard_txt}
        json.dump(j, f)


def historical_leaderboard(entries):
    """Keep historical threshold-based results separate from probability ROC AUC."""
    rows = []
    for entry in sorted(entries, key=lambda e: e["stored_scores"].balanced_accuracy.mean, reverse=True):
        scores = entry["stored_scores"]
        link = html.escape("../../" + entry["link"][:-3] + "/", quote=True)
        rows.append({
            "algorithm": f'<a href="{link}">{html.escape(entry["algorithm"])}</a>',
            "mean balanced accuracy": format_float(scores.balanced_accuracy.mean),
            "std balanced accuracy": format_float(scores.balanced_accuracy.std),
            "mean F1": format_float(scores.f1.mean),
        })
    table = pd.DataFrame(rows).to_html(index=False, escape=False) if rows else "<p>No historical submissions.</p>"
    return (
        '<details><summary>Historical balanced accuracy leaderboard</summary>\n'
        '<p>Stored threshold-based scores, including boolean-only submissions. '
        'These results are not probability ROC AUC and do not determine the current ranking. '
        'Original submission files are unchanged.</p>\n' + table + "\n</details>\n\n"
    )


def score_tables(scores, folds):
    """Render a consistent fold table and fold-statistics table."""
    names = list(scores)
    display = ["mape*" if name == "mape" else name for name in names]
    fold_table = "| fold | " + " | ".join(display) + " |\n" + "|------" * (len(names) + 1) + "|\n"
    for key, values in folds.items():
        fold_table += "| " + key + " | " + " | ".join(format_float(values[name]) for name in names) + " |\n"
    dist_table = "| metric | mean | max | min | std |\n|--------|------|-----|-----|-----|\n"
    for name, values in scores.items():
        label = "mape*" if name == "mape" else name
        dist_table += "| " + label + " | " + " | ".join(format_float(values[stat]) for stat in ("mean", "max", "min", "std")) + " |\n"
    return fold_table + "\n", dist_table + "\n"


def historical_score_tables(task):
    """Expose raw stored classification scores as an archive, not a current AUC claim."""
    names = list(task.scores)
    display = ["stored rocauc (historical)" if n == "rocauc" else n for n in names]
    folds = [{"fold": key, **{label: format_float(result.scores[name]) for name, label in zip(names, display)}}
             for key, result in task.results.items()]
    stats = [{"metric": label, **{stat: format_float(task.scores[name][stat]) for stat in ("mean", "max", "min", "std")}}
             for name, label in zip(names, display)]
    return (
        '<details><summary>Original stored classification scores (historical)</summary>\n'
        '<p>These values reproduce the original submission artifact. Older stored '
        '<code>rocauc</code> values were computed after thresholding and equal balanced accuracy; '
        'they must not be interpreted as the recomputed probability ROC AUC above.</p>\n'
        + pd.DataFrame(folds).to_html(index=False, escape=True)
        + pd.DataFrame(stats).to_html(index=False, escape=True) + "\n</details>\n\n"
    )


def generate_per_task_leaderboards(task_leaderboard_data_by_bmark):
    """
    Generate all .md files for the per-task leaderboards.


    Args:
        task_leaderboard_data_by_bmark:

    Returns:

    """
    for bmark_name, tasks_data in task_leaderboard_data_by_bmark.items():
        for task, entries in tasks_data.items():

            mbt = MatbenchTask(task, autoload=False, benchmark=bmark_name)
            task_type = mbt.metadata.task_type
            current_entries = [e for e in entries if e.get("eligible", True)]

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

            for entry in current_entries:
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
            # Break equal scores by algorithm name, matching homepage selection.
            df = df.sort_values(by=[sorting_column, "algorithm"],
                                ascending=[sorting_order, True], kind="mergesort")

            # header of the page
            header = f"# {bmark_name} {task}\n\n"

            subheader = f"## Individual Task Leaderboard for `{task}`\n\n"
            explanation = "_Leaderboard for an individual task. Algorithms shown here may include " \
                          "both general purpose and specialized algorithms (i.e., algorithms which " \
                          "are only valid for a subset of tasks in the benchmark._\n\n"
            if task_type == CLF_KEY:
                explanation += CLF_ROCAUC_NOTE
                explanation += "[Download classification score audit (JSON)](../static/classification_scores.json)\n\n"



            info_header = "### Dataset info\n\n"

            info_body = f"##### Description\n\n{mbt.metadata.description}\n\n"
            info_body += f"Number of samples: {mbt.metadata.n_samples}\n\n"
            info_body += f"Task type: {mbt.metadata.task_type}\n\n"
            info_body += f"Input type: {mbt.metadata.input_type}\n\n"
            info_body += f"##### Dataset columns\n\n" + "".join([f"- {c}: {cd}\n" for c, cd in mbt.metadata.columns.items()]) + "\n\n"
            info_body += f"##### Dataset reference\n\n `{mbt.metadata.reference}`\n\n"

            metadata_header = "### Metadata\n\n"
            metadata = f"```\n{pprint.pformat(mbt.metadata)}\n```\n\n"

            table_header = "### Probability ROC AUC leaderboard\n\n" if task_type == CLF_KEY else "### Leaderboard\n\n"
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


            if df.empty:
                table = "No eligible probability submissions are available for this task.\n\n" if task_type == CLF_KEY else "No submissions are available for this task.\n\n"
                fig_div = ""
            else:
                droppables = ["mean f1", "mean balanced_accuracy"] if task_type == CLF_KEY else ["mean rmse", "max max_error"]
                df = df.drop(columns=["algorithm w/ link"] + droppables)
                error_metric = "std rocauc" if task_type == CLF_KEY else "std mae"
                fig = px.scatter(df, error_y=error_metric, log_y=task_type == REG_KEY)

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
                                  xaxis_title="",
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  font={"color": "white"})
                fig.update_yaxes(linecolor="grey", gridcolor="grey")
                fig.update_xaxes(linecolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0)")

                fig_div = pio.to_html(fig, full_html=False, default_height="700px")

            history = historical_leaderboard(entries) if task_type == CLF_KEY else ""
            task_leaderboard_page = header + \
                                    subheader + \
                                    explanation + \
                                    table_header + \
                                    table + \
                                    fig_div + \
                                    history + \
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

    n_tasks = 0
    n_algos = 0
    n_benchmarks = 0

    gp_leaderboard_data_by_bmark = {}
    task_leaderboards_data_by_bmark = {}
    gp_graph_data_by_bmark = {}
    non_gp_graph_data_by_bmark = {}

    for bmark_name, bmarks in all_data_per_benchmark.items():
        n_benchmarks += 1
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
        non_gp_graph_data = copy.deepcopy(gp_graph_data)

        # A deterministic tie policy shared by the homepage and per-task tables.
        for bmark_data in sorted(bmarks, key=lambda packet: packet["info"]["algorithm"]):
            n_algos += 1
            mb = bmark_data["results"]
            info = bmark_data["info"]
            dir_name_short = bmark_data["dir_name_short"]

            for task in mb.tasks:
                n_tasks += 1
                task_name = task.dataset_name

                eligible = True
                scores = task.scores
                if task.metadata.task_type == REG_KEY:
                    score = scores.mae.mean

                    # Better regression tasks have lower mean mae
                    op = lt
                elif task.metadata.task_type == CLF_KEY:
                    classification = bmark_data["classification_scores"][task_name]
                    eligible = classification["eligible"]
                    scores = classification["scores"]
                    score = scores.rocauc.mean if eligible else None

                    # Better classification tasks have higher mean rocauc
                    op = gt
                else:
                    raise ValueError


                # Include both GP, structure-required, and structure/comp regression algos on
                # GP leaderboard, as there are 9 structure problems or 10 regression problems
                # across multiple dataset sizes (composition only or clf only are not included)
                if eligible and (mb.is_complete or mb.is_structure_complete or mb.is_regression_complete):
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

                if eligible:
                    non_gp_graph_data[task_name][info["algorithm"]] = score

                # Add it to the task-specific leaderboard, as all entries will be included
                # there
                task_leaderboards[task_name].append({
                    "scores": scores,
                    "stored_scores": task.scores,
                    "eligible": eligible,
                    "link": prefix + dir_name_short + ".md",
                    "algorithm": info["algorithm"],
                    "type": task.metadata.task_type
                })

            gp_leaderboard_data_by_bmark[bmark_name] = gp_leaderboard
            task_leaderboards_data_by_bmark[bmark_name] = task_leaderboards
            gp_graph_data_by_bmark[bmark_name] = gp_graph_data
            non_gp_graph_data_by_bmark[bmark_name] = non_gp_graph_data
    total_info_counts = {"benchmarks": n_benchmarks, "tasks": n_tasks, "algos": n_algos}

    return \
        gp_leaderboard_data_by_bmark, \
        task_leaderboards_data_by_bmark, \
        gp_graph_data_by_bmark, \
        non_gp_graph_data_by_bmark, \
        total_info_counts


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
                "Task type/input": f"{infod.task_type}/{infod.input_type}",
                "Target column (unit)": f"`{infod.target}` " + f"({infod.unit})" if infod.unit else f"`{infod.target}`",
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

        doc_str = generate_info_page(mb, info, dir_name_short, bmark_data["classification_scores"])

        doc_path = os.path.join(FULL_DATA_DIR, f"{dir_name_short}.md")
        with open(doc_path, "w", encoding="utf-8") as f:
            print(f"Writing full benchmark data page {doc_path}")
            f.write(doc_str)


def generate_info_page(mb: MatbenchBenchmark, info: dict, dir_name_short: str, classification_scores=None):
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

        score_note = ""
        history = ""
        task_scores = task.scores
        fold_scores = {key: result.scores for key, result in task.results.items()}
        if task.metadata.task_type == CLF_KEY:
            if classification_scores is None:
                raise ValueError("Classification scores must be prepared before generating detail pages")
            classification = classification_scores[task.dataset_name]
            if classification["eligible"]:
                task_scores = classification["scores"]
                fold_scores = classification["fold_scores"]
                score_note = (
                    "ROC AUC below is recomputed from stored floating-point probabilities "
                    "against the official test labels. Fold statistics use the same five folds "
                    "as the current leaderboard.\n\n"
                )
            else:
                task_scores = {k: v for k, v in task.scores.items() if k != "rocauc"}
                fold_scores = {key: {k: v for k, v in scores.items() if k != "rocauc"}
                               for key, scores in fold_scores.items()}
                score_note = (
                    "**Probability ROC AUC: unavailable.** This submission is excluded from "
                    "the current ROC AUC leaderboard. Exclusion reason: "
                    + classification["reason"].rstrip(".") + ".\n\n"
                )
            history = historical_score_tables(task)

        fold_table, dist_table = score_tables(task_scores, fold_scores)
        fold_data_header = "###### Fold scores\n\n"
        fold_dist_header = "###### Fold score stats\n\n"

        params_header = "###### Fold parameters\n\n"
        params_table = "| fold | params dict|\n" \
                       "|------|------------|\n"
        for fold_key, fold_data in task.results.items():
            params_short = f"{fold_data.parameters}"[:200]
            params_short = params_short + "..." if len(params_short) == 200 else params_short
            fold_line = f"| {fold_key} | `{params_short}` |\n"
            params_table += fold_line

        params_table += "\n\n"

        task_section = task_header + score_note + fold_data_header + fold_table + fold_dist_header + dist_table + history + params_header + params_table + "\n\n"
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

    prepare_classification_scores(all_data)
    nuke_docs(check=False)
    report_path = os.path.join(STATIC_DOCS_DIR, "classification_scores.json")
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(classification_score_report(all_data, BENCHMARKS_DIR), report_file,
                  indent=2, allow_nan=False)

    (
        gp_leaderboard_data_by_bmark,
        task_leaderboards_data_by_bmark,
        gp_graph_data_by_bmark,
        non_gp_graph_data_by_bmark,
        total_info_counts
    ) = organize_task_data(all_data)

    print("DOCS: ALL DATA ACQUIRED")

    generate_info_pages(all_data)

    generate_per_task_leaderboards(task_leaderboards_data_by_bmark)

    generate_metadata_pages(task_leaderboards_data_by_bmark)

    # must be called before generating the gp leaderboard
    gp_div = generate_scaled_errors_graph(
        gp_graph_data_by_bmark,
        output_fname=SCALED_ERRORS_PATH,
        title_txt="General Purpose Algorithms'",
        title_subtext="<br><sup>(only broadly applicable algorithms)</sup>"
    )
    non_gp_div = generate_scaled_errors_graph(
        non_gp_graph_data_by_bmark,
        output_fname=SCALED_ERRORS_NON_GP_PATH,
        title_txt="All Algorithms'",
        title_subtext="<br><sup>(includes task-specific algorithms)</sup>"
    )

    generate_general_purpose_leaderboard_and_plot(gp_leaderboard_data_by_bmark, total_info_counts, leaderboard_divs=[gp_div, non_gp_div])

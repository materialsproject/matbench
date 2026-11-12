import copy
import glob
import html
import json
import logging
import os
import pprint
import textwrap
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

# BA and probability ROC AUC share verified folds, but have different availability.
CLF_ROCAUC_NOTE = (
    '!!! info "Balanced accuracy and probability ROC AUC"\n'
    "    Both metrics are verified against official test labels, aligned by sample ID. "
    "Means and population standard deviations summarize the five official folds (ddof=0). "
    "Balanced accuracy (BA) uses p > 0.5 for saved probabilities and the saved decisions "
    "for historical hard labels, whose original thresholds may be unknown. Probability "
    "ROC AUC uses unthresholded positive-class probabilities. A dash (—) means no usable "
    "probabilities were saved, so probability ROC AUC cannot be recovered. "
    "Standard binary AUC on hard labels equals BA; probability ROC AUC is not "
    "necessarily higher than BA.\n\n"
)

CLF_HOMEPAGE_NOTE = (
    '!!! info "Classification best models: two distinct metrics"\n'
    "    Each classification task below shows the best balanced accuracy (BA) and the best "
    "probability ROC AUC, selected separately from the same general-purpose model scope. "
    "Every exact tie is listed. The two scaled-error plots continue to use BA only for "
    "classification. For all models and both metrics, see the "
    "[experimental metallicity](Leaderboards%20Per-Task/matbench_v0.1_matbench_expt_is_metal.md), "
    "[metallic glass](Leaderboards%20Per-Task/matbench_v0.1_matbench_glass.md), and "
    "[DFT metallicity](Leaderboards%20Per-Task/matbench_v0.1_matbench_mp_is_metal.md) task leaderboards. "
    "BA and probability ROC AUC can favor different models. Task leaderboards also include "
    "specialized models; the homepage general-purpose inclusion rules are unchanged.\n\n"
)

# Pointer to Matbench Discovery, which has its own leaderboard; see issue #388
MATBENCH_DISCOVERY_NOTE = "!!! tip \"Benchmarking ML for materials discovery?\"\n" \
                          "    For DFT-based materials discovery campaigns, use " \
                          "[Matbench Discovery](https://matbench-discovery.materialsproject.org/).\n\n"


# mp website statics locations
SCALED_ERRORS_PATH = os.path.join(STATIC_DOCS_DIR, "scaled_errors_{bmark_name}.json")
SCALED_ERRORS_NON_GP_PATH = os.path.join(STATIC_DOCS_DIR, "scaled_errors_non_gp_{bmark_name}.json")

pio.templates.default = "plotly_white"


def responsive_chart_html(fig, title, description, min_width=760, height=680):
    """Keep prose and a wrapping, keyboard-accessible legend outside Plotly.

    Only the plot canvas scrolls on narrow screens. Its minimum width preserves
    readable model/task labels rather than shrinking them until they overlap.
    """
    from uuid import uuid4

    chart_id = "matbench-chart-" + uuid4().hex
    legend = []
    for index, trace in enumerate(fig.data):
        color = trace.marker.color or "#a6a6a6"
        symbol = "diamond" if trace.marker.symbol == "diamond" else "circle"
        pressed = "false" if trace.visible in (False, "legendonly") else "true"
        legend.append(
            f'<button type="button" class="matbench-chart-key" data-trace="{index}" '
            f'aria-controls="{chart_id}" aria-pressed="{pressed}">'
            f'<span class="matbench-chart-swatch {symbol}" style="--trace-color:{html.escape(str(color), quote=True)}" '
            f'aria-hidden="true"></span><span>{html.escape(trace.name)}</span></button>'
        )
    fig.update_layout(title_text=None, showlegend=False, autosize=True)
    plot = pio.to_html(fig, full_html=False, div_id=chart_id,
                       default_height=f"{height}px", default_width="100%",
                       config={"responsive": True, "displaylogo": False})
    return (
        '<section class="matbench-chart">'
        f'<h3>{html.escape(title)}</h3><p class="matbench-chart-description">{description}</p>'
        '<div class="matbench-chart-legend" role="group" aria-label="Chart series; select a series to show or hide it">'
        + "".join(legend) + '</div>'
        '<p class="matbench-chart-scroll-hint">Scroll the chart horizontally to see every label. '
        'Select a legend item to show or hide its points.</p>'
        '<div class="matbench-chart-scroll" tabindex="0" role="region" '
        f'aria-label="{html.escape(title, quote=True)}; horizontally scrollable chart">'
        f'<div class="matbench-chart-canvas" style="min-width:{min_width}px">{plot}</div>'
        '</div></section>'
    )


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
                # BA=0 and BA=1 are valid scores, not missing-data sentinels.
                mask = np.isfinite(series) & series.ge(0.0) & series.le(1.0)
                mask_iix = np.where(mask)
                series.iloc[mask_iix] = (1 - series.iloc[mask_iix]) / 0.5
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
                        "Scaled Error: %{y}",
                        "Classification: (1 − balanced accuracy) / 0.5"
                    ]
                )
            )

            scale_note = " Linear scale includes zero scaled error." if not use_log_scale else ""
            scope_note = title_subtext.replace("<br>", "").replace("<sup>", "").replace("</sup>", "")
            chart_title = f"{title_txt} Scaled Errors"
            chart_description = (
                f"{html.escape(scope_note)} Lower is better. Classification: "
                "(1 − balanced accuracy) / 0.5. Regression: MAE / MAD."
                + scale_note
            )
            fig.update_layout(
                              meta={"classification_metric": "balanced_accuracy",
                                    "classification_scaled_error": "(1 - balanced_accuracy) / 0.5"},
                              yaxis_title="Scaled error (lower is better)",
                              xaxis_title="",
                              paper_bgcolor="rgba(0, 0, 0, 0)",
                              plot_bgcolor="rgba(0, 0, 0, 0)",
                              font={"color": "white"},
                              margin=dict(l=75, r=25, t=30, b=170))

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
                        'Algorithm: %{text}<br>Problem: %{x}<br>Scaled Error: %{y}<br>Classification: (1 − balanced accuracy) / 0.5<br>',
                    text=best_algos,
                    visible="legendonly",
                    name="Best algorithms"
                )
            )
            fig.update_traces(hoverlabel={"bgcolor": "black"}, selector=dict(type='scatter'))
            fig.update_xaxes(linecolor="grey", gridcolor="grey", tickangle=-40)
            fig.update_yaxes(linecolor="grey", gridcolor="grey")
            min_width = max(760, len(available.index) * 92 + 100)
            fig_div = responsive_chart_html(fig, chart_title, chart_description,
                                             min_width=min_width, height=680)


            if output_fname in (SCALED_ERRORS_PATH, SCALED_ERRORS_NON_GP_PATH):
                # Update layout and write files for showing on white background on mp website
                output_path = output_fname.format(bmark_name=bmark_name)
                fig.update_layout(
                    title_text="",
                    font={"color": "black"}, showlegend=True
                )
                print(f"Writing scaled errors graph to {output_path}")
                fig.write_json(output_path)
                export_html = responsive_chart_html(fig, chart_title, chart_description,
                                                      min_width=min_width, height=680)
                with open(output_path.replace(".json", ".html"), "w", encoding="utf-8") as output:
                    output.write(
                        '<!doctype html><html><head><meta charset="utf-8">'
                        '<meta name="viewport" content="width=device-width, initial-scale=1">'
                        f'<title>{html.escape(chart_title)}</title>'
                        '<link rel="stylesheet" href="../stylesheets/aesthetic_fixes.css">'
                        '</head><body style="margin:1rem;font-family:Arial,sans-serif;color:#202531">'
                        + export_html
                        + '<script src="../javascripts/responsive-plots.js"></script></body></html>'
                    )
            else:
                print(f"Output filename {output_fname} not required to write to disk.")

            return fig_div


def record_metric_winner(winners, metric, score, model):
    """Keep every exact top-score tie, with deterministic model-name ordering."""
    winner = winners[metric]
    if winner["score"] is None or score > winner["score"]:
        winner.update(score=score, models=[model])
    elif score == winner["score"]:
        winner["models"].append(model)
    winner["models"].sort(key=lambda item: item["algorithm"])


def homepage_metric_result(metric, winner, unit=None):
    """Render one clearly labelled metric and its independently selected models."""
    labels = {"balanced_accuracy": "Balanced accuracy (BA)",
              "rocauc": "Probability ROC AUC", "mae": "MAE"}
    label = labels[metric]
    if winner["models"]:
        models = []
        for model in winner["models"]:
            href = html.escape(model["link"] + "/", quote=True)
            name = html.escape(model["algorithm"])
            note = (' <span class="mb-model-note">(structure required)</span>'
                    if model["completeness"] == "structure" else "")
            models.append(f'<span class="mb-metric-model"><a href="{href}">{name}</a>{note}</span>')
        models_html = "; ".join(models)
        score = format_float(winner["score"])
        if metric == "mae" and unit:
            score += f" ({html.escape(unit)})"
        tie = '<span class="mb-metric-tie">Exact tie</span>' if len(models) > 1 else ""
    else:
        missing = ("No general-purpose submissions with verified balanced accuracy" if metric == "balanced_accuracy"
                   else "No general-purpose submissions with probability ROC AUC" if metric == "rocauc"
                   else "No general-purpose submissions")
        models_html = f'<span class="mb-metric-unavailable">{missing}</span>'
        score, tie = "—", ""
    return (
        f'<div class="mb-metric-result" data-metric="{metric}">'
        f'<span class="mb-metric-label">{label}</span> '
        f'<div class="mb-metric-models">{models_html} {tie}</div> '
        f'<strong class="mb-metric-score">{score}</strong></div>'
    )


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

        table_header = f"## Leaderboard-Property: General Purpose Algorithms on `{bmark}`\n\n"
        table_explanation = f"Find more information about this benchmark on [the benchmark info page]({METADATA_DIR_PREFIX}{bmark}.md)\n\n"
        table = (
            '<div class="mb-home-leaderboard-wrap">\n'
            '<table class="mb-home-leaderboard">\n'
            '<caption>Best results among general-purpose algorithms. Classification metrics are ranked separately: '
            'higher BA and probability ROC AUC are better; lower MAE is better.</caption>\n'
            '<thead><tr><th scope="col">Task</th><th scope="col">Samples</th>'
            '<th scope="col">Best results</th></tr></thead>\n<tbody>\n'
        )
        for task_name in sorted(gp_data, key=lambda name: metadata[name].n_samples):
            row = gp_data[task_name]
            task_metadata = metadata[task_name]
            task_link = html.escape(f"{PER_TASK_DIR_PREFIX}{bmark}_{task_name}/", quote=True)
            samples = format_int(task_metadata.n_samples)
            is_classification = task_metadata.task_type == CLF_KEY
            row_class = "mb-classification-row" if is_classification else "mb-regression-row"
            if is_classification:
                results = "".join(homepage_metric_result(metric, row["metric_winners"][metric])
                                  for metric in ("balanced_accuracy", "rocauc"))
            else:
                models = ([{key: row[key] for key in ("algorithm", "link", "completeness")}]
                          if row["algorithm"] is not None else [])
                results = homepage_metric_result("mae", {"score": row["score"], "models": models}, task_metadata.unit)
            table += (
                f'<tr class="mb-task-row {row_class}" data-task="{task_name}">'
                f'<th class="mb-task-name" scope="row" data-label="Task"><a href="{task_link}">{html.escape(task_name)}</a></th>'
                f'<td class="mb-samples" data-label="Samples">{samples}</td>'
                f'<td class="mb-best-results" data-label="Best results">{results}</td></tr>\n'
            )
        table += "</tbody>\n</table>\n</div>\n\n"
        # scaled_errors_plot_txt = f'\n<iframe src="static/{SCALED_ERRORS_FILENAME.format(bmark_name=bmark)}" class="is-fullwidth" height="1200px" width="1000px" frameBorder="0" allowtransparency="true"> </iframe>\n\n'
        # scaled_errors_non_gp_plt_txt = f'\n<iframe src="static/{SCALED_ERRORS_NON_GP_FILENAME.format(bmark_name=bmark)}" class="is-fullwidth" height="1200px" width="1000px" frameBorder="0"> </iframe>\n\n'
        audit_link = "[Download classification score audit (JSON)](./static/classification_scores.json)\n\n"
        gp_leaderboard_txt += table_header + table_explanation + CLF_HOMEPAGE_NOTE + audit_link + table

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


    final_txt = page_header + table_header + table_rows + gp_leaderboard_txt + static_txt_0 + "\n" + static_txt


    with open(os.path.join(DOCS_DIR, "index.md"), "w", encoding="utf-8") as f:
        print("Writing leaderboard and plot to index.md...")
        f.write(final_txt)

    with open(os.path.join(STATIC_DOCS_DIR,  "gp_table.json"), "w", encoding="utf-8") as f:
        print("Writing static gp_table json to gp_table.json...")
        j = {"txt": gp_leaderboard_txt,
             "classification_metric": "balanced_accuracy",
             "classification_scaled_error": "(1 - balanced_accuracy) / 0.5",
             "classification_metric_winners": {
                 bmark: {task: values["metric_winners"] for task, values in tasks.items()
                         if "metric_winners" in values}
                 for bmark, tasks in gp_leaderboard_data_by_bmark.items()},
             "classification_winner_scope": "General-purpose: complete, structure-complete, or regression-complete benchmarks; each metric requires its own verified score.",
             "general_purpose_winners": gp_leaderboard_data_by_bmark}
        json.dump(j, f)


def score_tables(scores, folds):
    """Render a consistent fold table and fold-statistics table."""
    names = [name for name, value in scores.items() if value is not None]
    labels = {"mape": "mape*", "rocauc": "probability ROC AUC",
              "balanced_accuracy": "balanced accuracy"}
    display = [labels.get(name, name) for name in names]
    fold_table = "| fold | " + " | ".join(display) + " |\n" + "|------" * (len(names) + 1) + "|\n"
    for key, values in folds.items():
        fold_table += "| " + key + " | " + " | ".join(format_float(values[name]) for name in names) + " |\n"
    dist_table = "| metric | mean | max | min | std |\n|--------|------|-----|-----|-----|\n"
    for name, values in scores.items():
        if values is None:
            continue
        label = labels.get(name, name)
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
        '<code>rocauc</code> values may reflect thresholded decisions, whose AUC equals balanced accuracy; '
        'they must not be interpreted as the recomputed probability ROC AUC above.</p>\n'
        + pd.DataFrame(folds).to_html(index=False, escape=True)
        + pd.DataFrame(stats).to_html(index=False, escape=True) + "\n</details>\n\n"
    )


def classification_leaderboard(entries, mbt):
    """Render two availability groups and both verified classification metrics.

    The ordering within each group uses that group's available ranking metric.
    There is intentionally no rank spanning the two groups.
    """
    valid = [entry for entry in entries if entry["ba_available"]]
    probability = sorted(
        (entry for entry in valid if entry["auc_available"]),
        key=lambda entry: (-entry["scores"].rocauc.mean, entry["algorithm"]),
    )
    hard_labels = sorted(
        (entry for entry in valid if not entry["auc_available"]),
        key=lambda entry: (-entry["scores"].balanced_accuracy.mean, entry["algorithm"]),
    )
    groups = [("ROC AUC available", probability), ("Balanced accuracy only", hard_labels)]
    table = (
        "The two sections group models by available metrics, not comparable ranks. "
        "The first is sorted by mean probability ROC AUC; the second by mean BA. "
        "Ties are ordered by algorithm name. All valid historical results are shown.\n\n"
    )
    columns = ["Algorithm", "Mean balanced accuracy", "Std balanced accuracy",
               "Mean probability ROC AUC", "Std probability ROC AUC", "Mean F1"]
    for name, group in groups:
        table += f"### {name}\n\n"
        if not group:
            table += "No verified submissions in this section.\n\n"
            continue
        table += "| " + " | ".join(columns) + " |\n" + "|------" * len(columns) + "|\n"
        for entry in group:
            scores = entry["scores"]
            ba = format_float(scores.balanced_accuracy.mean)
            auc = format_float(scores.rocauc.mean) if entry["auc_available"] else "—"
            auc_std = format_float(scores.rocauc.std) if entry["auc_available"] else "—"
            if entry["auc_available"]:
                auc = f"**{auc}**"
            else:
                ba = f"**{ba}**"
            table += (
                f"| [{entry['algorithm']}](../{entry['link']}) | {ba} | "
                f"{format_float(scores.balanced_accuracy.std)} | {auc} | {auc_std} | "
                f"{format_float(scores.f1.mean)} |\n"
            )
        table += "\n"

    excluded = [entry for entry in entries if not entry["ba_available"]]
    if excluded:
        table += "### Excluded invalid results\n\n"
        for entry in sorted(excluded, key=lambda entry: entry["algorithm"]):
            table += f"- {entry['algorithm']}: {entry['ba_reason']}\n"
        table += "\n"
    ordered = probability + hard_labels
    if not ordered:
        return table, "<p>No verified classification scores are available for this chart.</p>\n"

    fig = go.Figure()
    for metric, label, color, symbol, offset in (
        ("balanced_accuracy", "Balanced accuracy", "#a6a6a6", "circle", -0.12),
        ("rocauc", "Probability ROC AUC", "#329cff", "diamond", 0.12),
    ):
        selected = [(i, entry) for i, entry in enumerate(ordered)
                    if metric != "rocauc" or entry["auc_available"]]
        fig.add_trace(go.Scatter(
            name=label, mode="markers",
            x=[i + offset for i, _ in selected],
            y=[entry["scores"][metric].mean for _, entry in selected],
            error_y=dict(type="data", array=[entry["scores"][metric].std for _, entry in selected],
                         visible=True, color=color, thickness=1.5, width=4),
            marker=dict(color=color, symbol=symbol, size=10,
                        line=dict(color="#202531", width=0.6)),
            customdata=[[entry["algorithm"], entry["scores"][metric].std,
                         "ROC AUC available" if entry["auc_available"] else "Balanced accuracy only"]
                        for _, entry in selected],
            hovertemplate=(f"Algorithm: %{{customdata[0]}}<br>{label}: %{{y:.4f}}"
                           "<br>Fold std: %{customdata[1]:.4f}<br>Group: %{customdata[2]}<extra></extra>"),
        ))
    fig.update_layout(
        yaxis_title="Balanced accuracy / probability ROC AUC",
        xaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=70, r=25, t=55, b=170),
        meta=dict(classification_metrics=["balanced_accuracy", "rocauc"],
                  model_order=[entry["algorithm"] for entry in ordered],
                  auc_available_count=len(probability)),
    )
    # Mean ± fold standard deviation can extend outside the score's [0, 1]
    # domain. Keep the complete error bars as well as both valid endpoints.
    lower, upper = 0.0, 1.0
    for trace in fig.data:
        for mean, std in zip(trace.y, trace.error_y.array):
            lower = min(lower, mean - std)
            upper = max(upper, mean + std)
    fig.update_yaxes(type="linear", range=[lower - 0.04, upper + 0.04],
                     linecolor="grey", gridcolor="grey")
    fig.update_xaxes(
        type="linear", tickmode="array", tickvals=list(range(len(ordered))),
        ticktext=["<br>".join(textwrap.wrap(html.escape(entry["algorithm"]), width=22,
                                           break_long_words=False)) for entry in ordered],
        tickangle=-35, range=[-0.6, len(ordered) - 0.4],
        linecolor="rgba(0,0,0,0)", showgrid=False, zeroline=False,
    )
    if probability and hard_labels:
        boundary = len(probability) - 0.5
        fig.add_vline(x=boundary, line=dict(color="#b0b0b0", dash="dot", width=1))
    for name, group in groups:
        if group:
            start = 0 if name == "ROC AUC available" else len(probability)
            fig.add_annotation(x=start + (len(group) - 1) / 2, y=1.035,
                               xref="x", yref="paper", text=name, showarrow=False,
                               font=dict(size=12))
    chart = responsive_chart_html(
        fig, f"Balanced accuracy and probability ROC AUC: {mbt.metadata.target}",
        "Higher is better. Error bars show each metric's fold standard deviation. "
        "Models follow the tables: ROC AUC available first, then balanced accuracy only. "
        "The sections represent different available metrics, not comparable ranks.",
        min_width=max(760, len(ordered) * 96 + 95), height=640,
    )
    return table, chart


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
            if task_type == REG_KEY:
                rows = []
                for entry in entries:
                    if not entry.get("eligible", True):
                        continue
                    scores = entry["scores"]
                    rows.append({
                        "algorithm": entry["algorithm"],
                        "algorithm w/ link": f"[{entry['algorithm']}](../{entry['link']})",
                        "mean mae": scores.mae.mean,
                        "std mae": scores.mae.std,
                        "mean rmse": scores.rmse.mean,
                        "max max_error": scores.max_error.max,
                    })
                df = pd.DataFrame(rows, columns=["algorithm", "algorithm w/ link", "mean mae",
                                                "std mae", "mean rmse", "max max_error"])
                df = df.set_index("algorithm").sort_values(
                    by=["mean mae", "algorithm"], ascending=[True, True], kind="mergesort")
            elif task_type != CLF_KEY:
                raise ValueError(f"Task type {task_type} not recognized!")

            # header of the page
            header = f"# {bmark_name} {task}\n\n"

            subheader = f"## Individual Task Leaderboard for `{task}`\n\n"
            explanation = "_Leaderboard for an individual task. Algorithms shown here may include " \
                          "both general purpose and specialized algorithms (i.e., algorithms which " \
                          "are only valid for a subset of tasks in the benchmark._\n\n"
            if task_type == CLF_KEY:
                explanation += CLF_ROCAUC_NOTE
                explanation += (
                    "[Download classification score audit (JSON)](../static/classification_scores.json). "
                    "The audit preserves original cached scores, source paths and file hashes; "
                    "each algorithm's detail page also retains its historical score tables. "
                    "Original submission files are unchanged.\n\n"
                )



            info_header = "### Dataset info\n\n"

            info_body = f"##### Description\n\n{mbt.metadata.description}\n\n"
            info_body += f"Number of samples: {mbt.metadata.n_samples}\n\n"
            info_body += f"Task type: {mbt.metadata.task_type}\n\n"
            info_body += f"Input type: {mbt.metadata.input_type}\n\n"
            info_body += f"##### Dataset columns\n\n" + "".join([f"- {c}: {cd}\n" for c, cd in mbt.metadata.columns.items()]) + "\n\n"
            info_body += f"##### Dataset reference\n\n `{mbt.metadata.reference}`\n\n"

            metadata_header = "### Metadata\n\n"
            metadata = f"```\n{pprint.pformat(mbt.metadata)}\n```\n\n"

            table_header = "### Verified classification results\n\n" if task_type == CLF_KEY else "### Leaderboard\n\n"
            if task_type == CLF_KEY:
                table, fig_div = classification_leaderboard(entries, mbt)
            else:
                column_headers = df.columns
                table = "| " + " | ".join(column_headers) + " |\n" + "|------" * len(column_headers) + "|\n"
                for _, row in df.iterrows():
                    table += "| "
                    for column in column_headers:
                        if column == "algorithm w/ link":
                            table += f"{row[column]} | "
                        else:
                            number = format_float(row[column])
                            if column == "mean mae":
                                number = f"**{number}**"
                            table += f"{number} | "
                    table += "\n"
                table = table.replace("algorithm w/ link", "algorithm") + "\n"
                if df.empty:
                    table, fig_div = "No submissions are available for this task.\n\n", ""
                else:
                    df = df.drop(columns=["algorithm w/ link", "mean rmse", "max max_error"])
                    fig = px.scatter(df, error_y="std mae", log_y=True)
                    fig.update_layout(
                        title_text=f"Errors predicting '{mbt.metadata.target}' (lower is better)",
                        title_font_size=15, showlegend=False,
                        yaxis_title=f"MAE {mbt.metadata.unit}", xaxis_title="",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "white"})
                    fig.update_yaxes(linecolor="grey", gridcolor="grey")
                    fig.update_xaxes(linecolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0)")
                    fig_div = pio.to_html(fig, full_html=False, default_height="700px")

            task_leaderboard_page = header + \
                                    subheader + \
                                    explanation + \
                                    table_header + \
                                    table + \
                                    fig_div + \
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
        for task_name, task_metadata in metadata.items():
            if task_metadata.task_type == CLF_KEY:
                gp_leaderboard[task_name]["metric_winners"] = {
                    metric: {"score": None, "models": []}
                    for metric in ("balanced_accuracy", "rocauc")
                }

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
                ba_available = True
                classification = None
                scores = task.scores
                if task.metadata.task_type == REG_KEY:
                    score = scores.mae.mean

                    # Better regression tasks have lower mean mae
                    op = lt
                elif task.metadata.task_type == CLF_KEY:
                    classification = bmark_data["classification_scores"][task_name]
                    eligible = classification["eligible"]
                    ba_available = classification["ba_available"]
                    scores = classification["scores"]
                    score = scores.balanced_accuracy.mean if ba_available else None

                    # The homepage classification overview uses verified BA.
                    op = gt
                else:
                    raise ValueError


                # Include both GP, structure-required, and structure/comp regression algos on
                # GP leaderboard, as there are 9 structure problems or 10 regression problems
                # across multiple dataset sizes (composition only or clf only are not included)
                if ba_available and (mb.is_complete or mb.is_structure_complete or mb.is_regression_complete):
                    current_best_score = gp_leaderboard[task_name]["score"]

                    gp_graph_data[task_name][info["algorithm"]] = score
                    if classification is not None:
                        winners = gp_leaderboard[task_name]["metric_winners"]
                        model = {
                            "algorithm": info["algorithm"],
                            "link": prefix + dir_name_short,
                            "completeness": "all" if mb.is_complete or mb.is_regression_complete else "structure",
                        }
                        record_metric_winner(winners, "balanced_accuracy", score, model)
                        if classification["auc_available"]:
                            record_metric_winner(winners, "rocauc", scores.rocauc.mean, model.copy())

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

                if ba_available:
                    non_gp_graph_data[task_name][info["algorithm"]] = score

                # Add it to the task-specific leaderboard, as all entries will be included
                # there
                task_leaderboards[task_name].append({
                    "scores": scores,
                    "stored_scores": task.scores,
                    "eligible": eligible,
                    "ba_available": ba_available,
                    "auc_available": classification["auc_available"] if classification else None,
                    "ba_reason": classification["ba_reason"] if classification else None,
                    "auc_reason": classification["auc_reason"] if classification else None,
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
            if classification["ba_available"]:
                task_scores = classification["scores"]
                fold_scores = classification["fold_scores"]
                if classification["auc_available"]:
                    score_note = (
                        "Probability ROC AUC below is recomputed from the original saved "
                        "positive-class probabilities. Balanced accuracy, accuracy and F1 use "
                        "p > 0.5. Predictions are aligned by sample ID with official test labels. "
                        "Statistics are unweighted five-fold means and population standard "
                        "deviations (ddof=0).\n\n"
                    )
                else:
                    score_note = (
                        "**Probability ROC AUC: unavailable.** No usable probabilities were saved, "
                        "so probability ROC AUC cannot be recovered. Balanced accuracy, accuracy "
                        "and F1 below are recomputed from saved hard-label decisions aligned by "
                        "sample ID with official test labels. The original decision threshold may "
                        "be unknown; these historical decisions have not been reevaluated at a "
                        "common threshold. Statistics use unweighted five-fold means and "
                        "population standard deviations (ddof=0).\n\n"
                    )
            else:
                task_scores, fold_scores = {}, {}
                score_note = (
                    "**Verified classification metrics: unavailable.** This result is excluded "
                    "from both BA and probability ROC AUC displays. Reason: "
                    + classification["ba_reason"].rstrip(".") + ".\n\n"
                )
            score_note += "[Classification score audit and file provenance (JSON)](../static/classification_scores.json)\n\n"
            history = historical_score_tables(task)

        fold_table, dist_table = score_tables(task_scores, fold_scores) if task_scores else ("", "")
        fold_data_header = "###### Fold scores\n\n" if task_scores else ""
        fold_dist_header = "###### Fold score stats\n\n" if task_scores else ""

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

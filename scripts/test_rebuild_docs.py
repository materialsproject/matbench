"""Documentation integration tests. Run after installing requirements-docs.txt.

python -m unittest discover -s scripts -p 'test_rebuild_docs.py'
"""

import importlib.util
import json
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from matbench import metadata as metadata_module
from matbench.constants import CLF_METRICS, MBV01_KEY, REG_METRICS
from matbench.leaderboard import (
    classification_score_report,
    recompute_classification_scores,
)
from matbench.util import RecursiveDotDict

# The ordinary package test job does not need documentation-only dependencies.
if importlib.util.find_spec("plotly"):
    from scripts import rebuild_docs as docs
else:
    docs = None


def stats(values):
    return RecursiveDotDict(
        {
            name: {"mean": value, "max": value, "min": value, "std": 0.0}
            for name, value in values.items()
        }
    )


def rescore(result):
    task = result["results"].tasks[0]
    targets = {
        key: pd.Series([False, True], index=[f"{key}_negative", f"{key}_positive"])
        for key in task.folds_map.values()
    }
    result["classification_scores"][task.dataset_name] = (
        recompute_classification_scores(task, targets)
    )
    return result


def packet(algorithm, boolean=False, regression=False):
    name = "matbench_steels" if regression else "matbench_expt_is_metal"
    # Cached BA is deliberately stale. The actual Boolean BA is 1.0, while
    # the probabilities rank perfectly (AUC=1.0) but give BA=0.5 at p > 0.5.
    metrics = (
        {m: 0.25 for m in REG_METRICS}
        if regression
        else {m: 0.95 if boolean else 0.5 for m in CLF_METRICS}
    )
    task = SimpleNamespace(
        dataset_name=name,
        benchmark_name=MBV01_KEY,
        metadata=metadata_module.mbv01_metadata[name],
        scores=stats(metrics),
        folds_map={i: f"fold_{i}" for i in range(5)},
    )
    task.results = RecursiveDotDict(
        {
            key: {
                "data": (
                    {f"{key}_negative": False, f"{key}_positive": True}
                    if boolean
                    else {f"{key}_positive": 0.4, f"{key}_negative": 0.2}
                ),
                "scores": metrics.copy(),
                "parameters": {},
            }
            for key in task.folds_map.values()
        }
    )
    benchmark = SimpleNamespace(
        tasks=[task],
        benchmark_name=MBV01_KEY,
        metadata=metadata_module.mbv01_metadata,
        is_complete=True,
        is_structure_complete=True,
        is_composition_complete=True,
        is_regression_complete=True,
        is_classification_complete=True,
        user_metadata={},
    )
    info = dict(
        algorithm=algorithm,
        algorithm_long=algorithm,
        bibtex_refs=[],
        notes="",
        requirements={},
    )
    result = dict(
        results=benchmark,
        info=info,
        dir_name_short=algorithm,
        classification_scores={},
    )
    return result if regression else rescore(result)


class HomepageTableParser(HTMLParser):
    """Read metric labels, champions and links from the generated homepage."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = {}
        self.row = None
        self.metric = None
        self.metric_depth = 0

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag == "tr":
            self.row = (
                self.rows.setdefault(attributes["data-task"], {})
                if "data-task" in attributes
                else None
            )
        if self.row is not None:
            if "data-metric" in attributes:
                self.metric = self.row.setdefault(
                    attributes["data-metric"], {"text": "", "links": []}
                )
                self.metric_depth = 1
            elif self.metric is not None and tag not in ("br", "img", "input", "hr"):
                self.metric_depth += 1
            if self.metric is not None and tag == "a":
                self.metric["links"].append(attributes.get("href"))

    def handle_endtag(self, tag):
        if self.metric is not None and tag not in ("br", "img", "input", "hr"):
            self.metric_depth -= 1
            if self.metric_depth == 0:
                self.metric = None
        if tag == "tr":
            self.row = None

    def handle_data(self, text):
        if self.metric is not None:
            self.metric["text"] += text


@unittest.skipIf(
    docs is None, "Install requirements-docs.txt for documentation integration tests"
)
class TestClassificationLeaderboardDocs(unittest.TestCase):
    def setUp(self):
        self.float_packet = packet("Probability model")
        self.bool_packet = packet("Boolean model", boolean=True)
        self.reg_packet = packet("Regression model", regression=True)
        self.data = {
            "float": self.float_packet,
            "bool": self.bool_packet,
            "reg": self.reg_packet,
        }
        self.organized = docs.organize_task_data(self.data)
        self.task = "matbench_expt_is_metal"

    def task_page(self, entries, name=None):
        name = name or self.task
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            docs, "PER_TASK_DIR", tmp
        ), patch.object(docs.pio, "to_html", return_value="chart") as render:
            docs.generate_per_task_leaderboards({MBV01_KEY: {name: entries}})
            text = (Path(tmp) / f"{MBV01_KEY}_{name}.md").read_text()
            fig = render.call_args.args[0] if render.called else None
        return text, fig

    def homepage(self, gp, counts):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            docs, "DOCS_DIR", tmp
        ), patch.object(docs, "STATIC_DOCS_DIR", tmp):
            docs.generate_general_purpose_leaderboard_and_plot(gp, counts)
            text = (Path(tmp) / "index.md").read_text()
            data = json.loads((Path(tmp) / "gp_table.json").read_text())
        parser = HomepageTableParser()
        parser.feed(text)
        return text, data, parser.rows

    def assert_responsive_chart(self, text, fig):
        self.assertIn('<section class="matbench-chart">', text)
        self.assertIn("<h3>", text)
        self.assertIn('class="matbench-chart-description"', text)
        self.assertIn(
            'class="matbench-chart-scroll" tabindex="0" role="region"', text
        )
        self.assertIn('class="matbench-chart-canvas"', text)
        self.assertEqual(text.count('class="matbench-chart-key"'), len(fig.data))
        for index, trace in enumerate(fig.data):
            self.assertIn(f'data-trace="{index}"', text)
            self.assertIn(trace.name, text)
        self.assertEqual(text.count("aria-pressed="), len(fig.data))
        self.assertFalse(fig.layout.showlegend)
        self.assertIsNone(fig.layout.title.text)

    def test_homepage_ba_winner_can_differ_from_task_auc_winner(self):
        gp, entries, gp_graph, all_graph, counts = self.organized
        winner = gp[MBV01_KEY][self.task]
        self.assertEqual(winner["algorithm"], "Boolean model")
        self.assertEqual(winner["score"], 1.0)
        expected = {"Boolean model": 1.0, "Probability model": 0.5}
        self.assertEqual(gp_graph[MBV01_KEY][self.task], expected)
        self.assertEqual(all_graph[MBV01_KEY][self.task], expected)
        auc_entries = [
            e for e in entries[MBV01_KEY][self.task] if e["auc_available"]
        ]
        self.assertEqual(
            [e["algorithm"] for e in auc_entries], ["Probability model"]
        )
        self.assertEqual(auc_entries[0]["scores"].rocauc.mean, 1.0)
        self.assertEqual(counts["tasks"], 3)
        winners = winner["metric_winners"]
        self.assertEqual(winners["balanced_accuracy"]["score"], 1.0)
        self.assertEqual(winners["rocauc"]["score"], 1.0)
        self.assertEqual(
            [model["algorithm"] for model in winners["balanced_accuracy"]["models"]],
            ["Boolean model"],
        )
        self.assertEqual(
            [model["algorithm"] for model in winners["rocauc"]["models"]],
            ["Probability model"],
        )
        text, exported, rows = self.homepage(gp, counts)
        row = rows[self.task]
        ba, auc = row["balanced_accuracy"], row["rocauc"]
        self.assertIn("Boolean model", ba["text"])
        self.assertNotIn("Probability model", ba["text"])
        self.assertIn("Probability model", auc["text"])
        self.assertNotIn("Boolean model", auc["text"])
        self.assertIn("1.0000", ba["text"])
        self.assertIn("1.0000", auc["text"])
        self.assertIn("balanced accuracy", ba["text"].lower())
        self.assertIn("ROC AUC", auc["text"])
        self.assertEqual(len(ba["links"]), 1)
        self.assertEqual(len(auc["links"]), 1)
        self.assertIn("(./static/classification_scores.json)", text)
        self.assertEqual(
            exported["classification_metric_winners"][MBV01_KEY][self.task], winners
        )
        for graph in (gp_graph, all_graph):
            with patch.object(docs.pio, "to_html", return_value="chart") as render:
                docs.generate_scaled_errors_graph(graph, output_fname="not-written")
            fig = render.call_args.args[0]
            best = next(
                trace for trace in fig.data if trace.name == "Best algorithms"
            )
            self.assertIn("Boolean model", list(best.text))

    def test_homepage_lists_all_exact_metric_ties_in_stable_order(self):
        near_auc = packet("0 near AUC")
        near_auc["classification_scores"][self.task]["scores"].rocauc.mean = float(
            np.nextafter(1.0, 0.0)
        )
        near_ba = packet("0 near BA", boolean=True)
        near_ba["classification_scores"][self.task][
            "scores"
        ].balanced_accuracy.mean = float(np.nextafter(1.0, 0.0))
        items = [
            ("zp", packet("Z probability")),
            ("ap", packet("A probability")),
            ("zb", packet("Z labels", boolean=True)),
            ("ab", packet("A labels", boolean=True)),
            ("near_auc", near_auc),
            ("near_ba", near_ba),
        ]
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                gp, _, _, _, counts = docs.organize_task_data(
                    dict(items[::-1] if reverse else items)
                )
                winners = gp[MBV01_KEY][self.task]["metric_winners"]
                expected = {
                    "balanced_accuracy": ["A labels", "Z labels"],
                    "rocauc": ["A probability", "Z probability"],
                }
                _, exported, rows = self.homepage(gp, counts)
                for metric, algorithms in expected.items():
                    self.assertEqual(winners[metric]["score"], 1.0)
                    self.assertEqual(
                        [model["algorithm"] for model in winners[metric]["models"]],
                        algorithms,
                    )
                    rendered = rows[self.task][metric]
                    self.assertLess(
                        rendered["text"].index(algorithms[0]),
                        rendered["text"].index(algorithms[1]),
                    )
                    self.assertEqual(len(rendered["links"]), 2)
                    self.assertIn("tie", rendered["text"].lower())
                    self.assertNotIn("0 near", rendered["text"])
                self.assertEqual(
                    exported["classification_metric_winners"][MBV01_KEY][self.task],
                    winners,
                )

    def test_homepage_missing_auc_and_real_auc_boundaries_are_distinct(self):
        for auc_value in (None, 0.0, 1.0):
            with self.subTest(auc=auc_value):
                data = {"labels": self.bool_packet}
                if auc_value is not None:
                    p = packet("Probability model")
                    if auc_value == 0.0:
                        for key, fold in p["results"].tasks[0].results.items():
                            fold.data = {
                                f"{key}_negative": 0.8,
                                f"{key}_positive": 0.2,
                            }
                        rescore(p)
                    data["probabilities"] = p
                gp, _, _, _, counts = docs.organize_task_data(data)
                winner = gp[MBV01_KEY][self.task]["metric_winners"]["rocauc"]
                _, exported, rows = self.homepage(gp, counts)
                rendered = rows[self.task]["rocauc"]
                self.assertEqual(winner["score"], auc_value)
                if auc_value is None:
                    self.assertEqual(winner["models"], [])
                    self.assertIn("—", rendered["text"])
                    self.assertNotIn("0.0000", rendered["text"])
                    self.assertEqual(rendered["links"], [])
                else:
                    self.assertEqual(
                        [model["algorithm"] for model in winner["models"]],
                        ["Probability model"],
                    )
                    self.assertIn(f"{auc_value:.4f}", rendered["text"])
                    self.assertNotIn("—", rendered["text"])
                    self.assertEqual(len(rendered["links"]), 1)
                self.assertEqual(
                    exported["classification_metric_winners"][MBV01_KEY][self.task][
                        "rocauc"
                    ],
                    winner,
                )

    def test_audit_distinguishes_homepage_table_metrics_from_scaled_error_metric(
        self,
    ):
        data = {"probabilities": self.float_packet, "labels": self.bool_packet}
        with tempfile.TemporaryDirectory() as tmp:
            for p in data.values():
                artifact = Path(tmp) / p["dir_name_short"] / "results.json.gz"
                artifact.parent.mkdir()
                artifact.write_bytes(b"immutable source artifact fixture")
            audit = classification_score_report(data, tmp)
        self.assertEqual(
            audit["homepage_table_metrics"], ["balanced_accuracy", "rocauc"]
        )
        self.assertEqual(audit["homepage_scaled_error_metric"], "balanced_accuracy")
        self.assertIn("general-purpose", audit["homepage_scope"])
        self.assertNotIn("homepage_metric", audit)

    def test_main_table_shows_both_groups_and_missing_auc_explicitly(self):
        text, _ = self.task_page(self.organized[1][MBV01_KEY][self.task])
        current = text
        auc_group, ba_group = current.split("### Balanced accuracy only", 1)
        self.assertIn("### ROC AUC available", auc_group)
        self.assertIn("Probability model", auc_group)
        self.assertIn("Boolean model", ba_group)
        self.assertIn("balanced accuracy", current.lower())
        self.assertIn("probability ROC AUC", current)
        bool_row = next(
            line
            for line in ba_group.splitlines()
            if line.startswith("| [Boolean model]")
        )
        self.assertIn("1.0000", bool_row)
        self.assertIn("—", bool_row)
        self.assertNotIn("0.9500", bool_row)
        self.assertIn("(../static/classification_scores.json)", current)
        self.assertNotIn("Original cached scores and provenance", text)
        self.assertNotIn("<details>", text)
        self.assertEqual(text.count("| [Boolean model]"), 1)
        self.assertEqual(text.count("| [Probability model]"), 1)

    def test_details_use_recomputed_folds_and_preserve_original_results(self):
        for p in (self.float_packet, self.bool_packet):
            with self.subTest(algorithm=p["info"]["algorithm"]):
                original = json.loads(json.dumps(p["results"].tasks[0].results))
                text = docs.generate_info_page(
                    p["results"],
                    p["info"],
                    p["dir_name_short"],
                    p["classification_scores"],
                )
                current, history = text.split("<details>", 1)
                self.assertIn("stored rocauc (historical)", history)
                self.assertEqual(p["results"].tasks[0].results, original)
                if p is self.float_packet:
                    self.assertIn(
                        "| probability ROC AUC | 1.0000 | 1.0000 | "
                        "1.0000 | 0.0000 |",
                        current,
                    )
                    self.assertIn(
                        "| balanced accuracy | 0.5000 | 0.5000 | 0.5000 | 0.0000 |",
                        current,
                    )
                else:
                    self.assertIn("Probability ROC AUC: unavailable", current)
                    self.assertNotIn("| probability ROC AUC |", current)
                    self.assertIn(
                        "| balanced accuracy | 1.0000 | 1.0000 | 1.0000 | 0.0000 |",
                        current,
                    )
                    self.assertNotIn("0.9500", current)

    def test_dual_plot_has_all_ba_and_separate_auc_fold_error_bars(self):
        p = packet("Probability model")
        for index, (key, fold) in enumerate(p["results"].tasks[0].results.items()):
            fold.data = {
                f"{key}_negative": 0.2,
                f"{key}_positive": 0.4 if index < 2 else 0.8,
            }
        rescore(p)
        entries = docs.organize_task_data({"float": p, "bool": self.bool_packet})[1][
            MBV01_KEY
        ][self.task]
        text, fig = self.task_page(entries)
        traces = {trace.name: trace for trace in fig.data}
        ba = traces["Balanced accuracy"]
        auc = traces["Probability ROC AUC"]
        self.assertEqual(list(ba.y), [0.8, 1.0])
        self.assertEqual(list(auc.y), [1.0])
        np.testing.assert_allclose(
            ba.error_y.array, [np.std([0.5, 0.5, 1, 1, 1], ddof=0), 0.0]
        )
        self.assertEqual(list(auc.error_y.array), [0.0])
        self.assertNotEqual(ba.marker.color, auc.marker.color)
        self.assertNotEqual(ba.marker.symbol, auc.marker.symbol)
        self.assertNotEqual(ba.x[0], auc.x[0])
        self.assertEqual(
            list(fig.layout.xaxis.ticktext), ["Probability model", "Boolean model"]
        )
        self.assertNotEqual(fig.layout.yaxis.type, "log")
        self.assertIn("balanced accuracy", fig.layout.yaxis.title.text.lower())
        self.assertIn("ROC AUC", fig.layout.yaxis.title.text)
        self.assertIn("Balanced accuracy only", text)
        self.assert_responsive_chart(text, fig)

    def test_classification_plot_does_not_clip_fold_standard_deviations(self):
        high = packet("High mean")
        low = packet("Low mean")
        for result, correct_folds in ((high, 4), (low, 1)):
            for index, (key, fold) in enumerate(
                result["results"].tasks[0].results.items()
            ):
                negative, positive = (
                    (0.2, 0.8) if index < correct_folds else (0.8, 0.2)
                )
                fold.data = {
                    f"{key}_negative": negative,
                    f"{key}_positive": positive,
                }
            rescore(result)
        entries = docs.organize_task_data({"high": high, "low": low})[1][MBV01_KEY][
            self.task
        ]
        _, fig = self.task_page(entries)
        traces = {trace.name: trace for trace in fig.data}
        # The five fold scores [1, 1, 1, 1, 0] have mean=.8 and std=.4.
        # A valid dispersion bar can extend beyond the score's [0, 1] domain.
        for metric in ("Balanced accuracy", "Probability ROC AUC"):
            trace = traces[metric]
            np.testing.assert_allclose(trace.y, [0.8, 0.2])
            np.testing.assert_allclose(trace.error_y.array, [0.4, 0.4])
            self.assertGreaterEqual(
                fig.layout.yaxis.range[1],
                max(np.array(trace.y) + trace.error_y.array),
            )
            self.assertLessEqual(
                fig.layout.yaxis.range[0],
                min(np.array(trace.y) - trace.error_y.array),
            )
        self.assertNotEqual(fig.layout.yaxis.type, "log")
        self.assertLessEqual(fig.layout.yaxis.range[0], 0.0)
        self.assertGreaterEqual(fig.layout.yaxis.range[1], 1.0)

    def test_zero_and_one_auc_are_scores_not_missing_values(self):
        reversed_p = packet("AUC zero")
        for key, fold in reversed_p["results"].tasks[0].results.items():
            fold.data = {f"{key}_negative": 0.8, f"{key}_positive": 0.2}
        rescore(reversed_p)
        entries = docs.organize_task_data(
            {"zero": reversed_p, "one": self.float_packet, "bool": self.bool_packet}
        )[1][MBV01_KEY][self.task]
        text, fig = self.task_page(entries)
        traces = {trace.name: trace for trace in fig.data}
        self.assertEqual(list(traces["Probability ROC AUC"].y), [1.0, 0.0])
        self.assertEqual(list(traces["Balanced accuracy"].y), [0.5, 0.0, 1.0])
        self.assertNotEqual(fig.layout.yaxis.type, "log")
        self.assertLessEqual(fig.layout.yaxis.range[0], 0.0)
        self.assertGreaterEqual(fig.layout.yaxis.range[1], 1.0)
        current = text.split("<details>", 1)[0]
        zero_row = next(
            line for line in current.splitlines() if line.startswith("| [AUC zero]")
        )
        self.assertIn("0.0000", zero_row)
        self.assertNotIn("—", zero_row)
        bool_row = next(
            line
            for line in current.splitlines()
            if line.startswith("| [Boolean model]")
        )
        self.assertIn("—", bool_row)

    def test_group_and_tie_order_is_stable_in_table_and_plot(self):
        lower_auc = packet("Lower AUC probability")
        lower_auc["results"].tasks[0].results["fold_0"].data = {
            "fold_0_negative": 0.8,
            "fold_0_positive": 0.2,
        }
        rescore(lower_auc)
        lower_ba = packet("Lower BA labels", boolean=True)
        lower_ba["results"].tasks[0].results["fold_0"].data = {
            "fold_0_negative": True,
            "fold_0_positive": False,
        }
        rescore(lower_ba)
        items = [
            ("zp", packet("Z probability")),
            ("ap", packet("A probability")),
            ("zb", packet("Z labels", boolean=True)),
            ("ab", packet("A labels", boolean=True)),
            ("low_auc", lower_auc),
            ("low_ba", lower_ba),
        ]
        expected = [
            "A probability",
            "Z probability",
            "Lower AUC probability",
            "A labels",
            "Z labels",
            "Lower BA labels",
        ]
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                selected = items[::-1] if reverse else items
                gp, entries, _, _, _ = docs.organize_task_data(dict(selected))
                self.assertEqual(gp[MBV01_KEY][self.task]["algorithm"], "A labels")
                # The table and chart must sort correctly even if input is reversed.
                text, fig = self.task_page(entries[MBV01_KEY][self.task][::-1])
                current = text.split("<details>", 1)[0]
                positions = [current.index(f"| [{name}]") for name in expected]
                self.assertEqual(positions, sorted(positions))
                self.assertEqual(list(fig.layout.xaxis.ticktext), expected)

    def test_ba_only_task_remains_visible_without_fabricating_auc(self):
        gp, entries, gp_graph, _, counts = docs.organize_task_data(
            {"bool": self.bool_packet}
        )
        self.assertEqual(gp[MBV01_KEY][self.task]["algorithm"], "Boolean model")
        self.assertEqual(gp_graph[MBV01_KEY][self.task], {"Boolean model": 1.0})
        text, fig = self.task_page(entries[MBV01_KEY][self.task])
        current = text.split("<details>", 1)[0]
        self.assertIn("Boolean model", current)
        self.assertIn("Balanced accuracy only", current)
        self.assertEqual(
            list(
                next(
                    trace for trace in fig.data if trace.name == "Balanced accuracy"
                ).y
            ),
            [1.0],
        )
        auc = [trace for trace in fig.data if trace.name == "Probability ROC AUC"]
        self.assertTrue(not auc or len(auc[0].y) == 0)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            docs, "DOCS_DIR", tmp
        ), patch.object(docs, "STATIC_DOCS_DIR", tmp):
            docs.generate_general_purpose_leaderboard_and_plot(gp, counts)
            self.assertIn("Boolean model", (Path(tmp) / "index.md").read_text())

    def test_invalid_predictions_cannot_enter_through_ba_only_group(self):
        invalid = packet("Invalid model", boolean=True)
        invalid["results"].tasks[0].results["fold_0"].data["fold_0_negative"] = 0.2
        rescore(invalid)
        gp, entries, gp_graph, all_graph, _ = docs.organize_task_data(
            {"invalid": invalid, "float": self.float_packet}
        )
        self.assertEqual(gp[MBV01_KEY][self.task]["algorithm"], "Probability model")
        self.assertNotIn("Invalid model", gp_graph[MBV01_KEY][self.task])
        self.assertNotIn("Invalid model", all_graph[MBV01_KEY][self.task])
        text, fig = self.task_page(entries[MBV01_KEY][self.task])
        current = text.split("<details>", 1)[0]
        self.assertNotIn("| [Invalid model]", current)
        self.assertNotIn("Invalid model", list(fig.layout.xaxis.ticktext))

    def test_homepage_preserves_general_purpose_inclusion_scope(self):
        specialized = packet("Specialist", boolean=True)
        benchmark = specialized["results"]
        benchmark.is_complete = benchmark.is_structure_complete = (
            benchmark.is_regression_complete
        ) = False
        gp, entries, gp_graph, all_graph, _ = docs.organize_task_data(
            {"special": specialized, "float": self.float_packet}
        )
        self.assertEqual(gp[MBV01_KEY][self.task]["algorithm"], "Probability model")
        self.assertNotIn("Specialist", gp_graph[MBV01_KEY][self.task])
        self.assertEqual(all_graph[MBV01_KEY][self.task]["Specialist"], 1.0)
        self.assertIn(
            "Specialist",
            [entry["algorithm"] for entry in entries[MBV01_KEY][self.task]],
        )

    def test_homepage_auc_champion_uses_the_same_general_purpose_scope_as_ba(self):
        specialist = packet("Specialist probability")
        specialist_benchmark = specialist["results"]
        specialist_benchmark.is_complete = (
            specialist_benchmark.is_structure_complete
        ) = specialist_benchmark.is_regression_complete = False
        general = packet("General probability")
        for key, fold in general["results"].tasks[0].results.items():
            fold.data = {f"{key}_negative": 0.8, f"{key}_positive": 0.2}
        rescore(general)
        data = {
            "specialist": specialist,
            "general": general,
            "labels": self.bool_packet,
        }
        gp, entries, gp_graph, all_graph, counts = docs.organize_task_data(data)
        winners = gp[MBV01_KEY][self.task]["metric_winners"]
        self.assertEqual(
            [model["algorithm"] for model in winners["balanced_accuracy"]["models"]],
            ["Boolean model"],
        )
        self.assertEqual(
            [model["algorithm"] for model in winners["rocauc"]["models"]],
            ["General probability"],
        )
        self.assertEqual(winners["rocauc"]["score"], 0.0)
        self.assertNotIn("Specialist probability", gp_graph[MBV01_KEY][self.task])
        self.assertIn("Specialist probability", all_graph[MBV01_KEY][self.task])
        _, _, rows = self.homepage(gp, counts)
        self.assertNotIn("Specialist probability", rows[self.task]["rocauc"]["text"])
        _, task_fig = self.task_page(entries[MBV01_KEY][self.task])
        self.assertEqual(
            list(task_fig.layout.xaxis.ticktext)[0], "Specialist probability"
        )

    def test_scaled_chart_uses_ba_boundaries_without_fake_epsilon(self):
        graph = {
            MBV01_KEY: {self.task: {"BA zero": 0.0, "BA half": 0.5, "BA one": 1.0}}
        }
        with patch.object(docs.pio, "to_html", return_value="chart") as render:
            chart_html = docs.generate_scaled_errors_graph(
                graph, output_fname="not-written"
            )
            fig = render.call_args.args[0]
        self.assertEqual(fig.layout.yaxis.type, "linear")
        values = {
            trace.name: list(trace.y)
            for trace in fig.data
            if trace.name != "Best algorithms"
        }
        self.assertEqual(
            values, {"BA zero": [2.0], "BA half": [1.0], "BA one": [0.0]}
        )
        self.assertIn("zero scaled error", chart_html)
        self.assert_responsive_chart(chart_html, fig)
        graph = {MBV01_KEY: {self.task: {"BA zero": 0.0, "BA half": 0.5}}}
        with patch.object(docs.pio, "to_html", return_value="chart") as render:
            docs.generate_scaled_errors_graph(graph, output_fname="not-written")
            self.assertEqual(render.call_args.args[0].layout.yaxis.type, "log")

    def test_regression_scores_details_sorting_and_axis_are_unchanged(self):
        gp, entries, gp_graph, _, counts = self.organized
        name = "matbench_steels"
        expected = self.reg_packet["results"].tasks[0].scores
        self.assertEqual(gp[MBV01_KEY][name]["score"], expected.mae.mean)
        self.assertIs(entries[MBV01_KEY][name][0]["scores"], expected)
        self.assertEqual(gp_graph[MBV01_KEY][name], {"Regression model": 0.25})
        _, exported, homepage_rows = self.homepage(gp, counts)
        self.assertEqual(set(homepage_rows[name]), {"mae"})
        self.assertIn("Regression model", homepage_rows[name]["mae"]["text"])
        self.assertIn("0.2500", homepage_rows[name]["mae"]["text"])
        self.assertNotIn(name, exported["classification_metric_winners"][MBV01_KEY])
        p = self.reg_packet
        text = docs.generate_info_page(
            p["results"], p["info"], p["dir_name_short"], p["classification_scores"]
        )
        self.assertIn("| mae | 0.2500 | 0.2500 | 0.2500 | 0.0000 |", text)
        self.assertNotIn("Probability ROC AUC", text)
        self.assertNotIn("<details>", text)
        _, fig = self.task_page(entries[MBV01_KEY][name], name=name)
        self.assertEqual(fig.layout.yaxis.type, "log")
        for reverse in (False, True):
            items = [
                ("z", packet("Z model", regression=True)),
                ("a", packet("A model", regression=True)),
            ]
            gp, entries, graph, _, _ = docs.organize_task_data(
                dict(items[::-1] if reverse else items)
            )
            self.assertEqual(gp[MBV01_KEY][name]["algorithm"], "A model")
            text, _ = self.task_page(entries[MBV01_KEY][name][::-1], name=name)
            self.assertLess(text.index("| [A model]"), text.index("| [Z model]"))
            with patch.object(docs.pio, "to_html", return_value="chart") as render:
                docs.generate_scaled_errors_graph(graph, output_fname="not-written")
            best = next(
                trace
                for trace in render.call_args.args[0].data
                if trace.name == "Best algorithms"
            )
            self.assertEqual(list(best.text), ["A model"])


if __name__ == "__main__":
    unittest.main()

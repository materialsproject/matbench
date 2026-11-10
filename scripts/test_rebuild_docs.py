"""Documentation integration tests. Run after installing requirements-docs.txt.

python -m unittest discover -s scripts -p 'test_rebuild_docs.py'
"""
import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from matbench.constants import CLF_KEY, CLF_METRICS, MBV01_KEY, REG_KEY, REG_METRICS
from matbench.leaderboard import recompute_classification_scores
from matbench import metadata as metadata_module
from matbench.util import RecursiveDotDict

# The ordinary package test job does not need documentation-only dependencies.
if importlib.util.find_spec("plotly"):
    from scripts import rebuild_docs as docs
else:
    docs = None


def stats(values):
    return RecursiveDotDict({name: {"mean": value, "max": value, "min": value, "std": 0.0}
                             for name, value in values.items()})


def packet(algorithm, boolean=False, regression=False):
    name = "matbench_steels" if regression else "matbench_expt_is_metal"
    metrics = {m: 0.25 for m in REG_METRICS} if regression else {m: 0.95 if boolean else 0.5 for m in CLF_METRICS}
    task = SimpleNamespace(dataset_name=name, benchmark_name=MBV01_KEY,
                           metadata=metadata_module.mbv01_metadata[name], scores=stats(metrics),
                           folds_map={i: f"fold_{i}" for i in range(5)})
    task.results = RecursiveDotDict({key: {"data": {"negative": False, "positive": True} if boolean else
                                          {"positive": 0.4, "negative": 0.2},
                                          "scores": metrics.copy(), "parameters": {}}
                                    for key in task.folds_map.values()})
    benchmark = SimpleNamespace(tasks=[task], benchmark_name=MBV01_KEY, metadata=metadata_module.mbv01_metadata,
                                is_complete=True, is_structure_complete=True,
                                is_composition_complete=True, is_regression_complete=True,
                                is_classification_complete=True, user_metadata={})
    info = dict(algorithm=algorithm, algorithm_long=algorithm, bibtex_refs=[], notes="", requirements={})
    result = dict(results=benchmark, info=info, dir_name_short=algorithm, classification_scores={})
    if not regression:
        targets = {key: pd.Series([False, True], index=["negative", "positive"])
                   for key in task.folds_map.values()}
        result["classification_scores"][name] = recompute_classification_scores(task, targets)
    return result


@unittest.skipIf(docs is None, "Install requirements-docs.txt for documentation integration tests")
class TestProbabilityLeaderboardDocs(unittest.TestCase):
    def setUp(self):
        self.float_packet = packet("Probability model")
        self.bool_packet = packet("Boolean model", boolean=True)
        self.reg_packet = packet("Regression model", regression=True)
        self.data = {"float": self.float_packet, "bool": self.bool_packet, "reg": self.reg_packet}
        self.organized = docs.organize_task_data(self.data)
        self.task = "matbench_expt_is_metal"

    def test_homepage_winner_and_graphs_use_same_probability_scores(self):
        gp, entries, gp_graph, all_graph, counts = self.organized
        winner = gp[MBV01_KEY][self.task]
        self.assertEqual(winner["algorithm"], "Probability model")
        self.assertEqual(winner["score"], 1.0)
        current = [e for e in entries[MBV01_KEY][self.task] if e["eligible"]]
        self.assertEqual([e["algorithm"] for e in current], [winner["algorithm"]])
        self.assertEqual(current[0]["scores"].rocauc.mean, winner["score"])
        self.assertEqual(gp_graph[MBV01_KEY][self.task], {"Probability model": 1.0})
        self.assertEqual(all_graph[MBV01_KEY][self.task], {"Probability model": 1.0})
        # Reporting eligibility must not pretend a submitted algorithm disappeared.
        self.assertEqual(counts["tasks"], 3)
        with tempfile.TemporaryDirectory() as tmp, patch.object(docs, "DOCS_DIR", tmp), patch.object(docs, "STATIC_DOCS_DIR", tmp):
            docs.generate_general_purpose_leaderboard_and_plot(gp, counts)
            text = (Path(tmp) / "index.md").read_text()
            line = next(line for line in text.splitlines() if line.startswith("| [matbench_expt_is_metal]"))
            self.assertIn("[Probability model]", line)
            self.assertIn("**1.0000**", line)
            self.assertNotIn("Boolean model", line)
            self.assertIn("(./static/classification_scores.json)", text)

    def test_current_table_and_collapsed_history_are_distinct(self):
        entries = self.organized[1]
        with tempfile.TemporaryDirectory() as tmp, patch.object(docs, "PER_TASK_DIR", tmp), patch.object(docs.pio, "to_html", return_value="<div>chart</div>"):
            docs.generate_per_task_leaderboards({MBV01_KEY: {self.task: entries[MBV01_KEY][self.task]}})
            text = (Path(tmp) / f"{MBV01_KEY}_{self.task}.md").read_text()
        current, history = text.split("<details>", 1)
        self.assertIn("**1.0000**", current)
        self.assertIn("Probability model", current)
        self.assertIn("(../static/classification_scores.json)", current)
        self.assertNotIn("Boolean model", current)
        self.assertIn("Historical balanced accuracy leaderboard", history)
        self.assertIn("Boolean model", history)
        self.assertIn("mean balanced accuracy", history)
        self.assertNotIn("mean rocauc", history)
        self.assertIn('href="../../Full%20Benchmark%20Data/Boolean model/"', history)

    def test_details_use_recomputed_folds_but_preserve_original_artifact(self):
        p = self.float_packet
        original = json.loads(json.dumps(p["results"].tasks[0].results))
        text = docs.generate_info_page(p["results"], p["info"], p["dir_name_short"], p["classification_scores"])
        current, history = text.split("<details>", 1)
        self.assertIn("| rocauc | 1.0000 | 1.0000 | 1.0000 | 0.0000 |", current)
        self.assertIn("stored rocauc (historical)", history)
        self.assertEqual(p["results"].tasks[0].results, original)
        p = self.bool_packet
        text = docs.generate_info_page(p["results"], p["info"], p["dir_name_short"], p["classification_scores"])
        current, history = text.split("<details>", 1)
        self.assertIn("Probability ROC AUC: unavailable", current)
        self.assertNotIn("| rocauc |", current)
        self.assertIn("stored rocauc (historical)", history)

    def test_regression_scores_and_details_are_unchanged(self):
        gp, entries, gp_graph, all_graph, _ = self.organized
        name = "matbench_steels"
        expected = self.reg_packet["results"].tasks[0].scores
        self.assertEqual(gp[MBV01_KEY][name]["score"], expected.mae.mean)
        self.assertIs(entries[MBV01_KEY][name][0]["scores"], expected)
        self.assertEqual(gp_graph[MBV01_KEY][name], {"Regression model": 0.25})
        p = self.reg_packet
        text = docs.generate_info_page(p["results"], p["info"], p["dir_name_short"], p["classification_scores"])
        self.assertIn("| mae | 0.2500 | 0.2500 | 0.2500 | 0.0000 |", text)
        self.assertNotIn("Probability ROC AUC", text)
        self.assertNotIn("<details>", text)

    def test_classification_plot_keeps_auc_zero_on_linear_axis(self):
        p = packet("Reversed probability model")
        task = p["results"].tasks[0]
        for fold in task.results.values():
            fold.data = {"negative": 0.8, "positive": 0.2}
        targets = {key: pd.Series([False, True], index=["negative", "positive"])
                   for key in task.folds_map.values()}
        p["classification_scores"][self.task] = recompute_classification_scores(task, targets)
        entries = docs.organize_task_data({"reversed": p})[1]
        with tempfile.TemporaryDirectory() as tmp, patch.object(docs, "PER_TASK_DIR", tmp), patch.object(docs.pio, "to_html", return_value="chart") as render:
            docs.generate_per_task_leaderboards({MBV01_KEY: {self.task: entries[MBV01_KEY][self.task]}})
            fig = render.call_args.args[0]
            self.assertNotEqual(fig.layout.yaxis.type, "log")
            self.assertEqual(list(fig.data[0].y), [0.0])
            text = (Path(tmp) / f"{MBV01_KEY}_{self.task}.md").read_text()
            self.assertIn("**0.0000**", text.split("<details>", 1)[0])
        # Regression charts keep their established logarithmic axis.
        entries = self.organized[1]
        with tempfile.TemporaryDirectory() as tmp, patch.object(docs, "PER_TASK_DIR", tmp), patch.object(docs.pio, "to_html", return_value="chart") as render:
            docs.generate_per_task_leaderboards({MBV01_KEY: {"matbench_steels": entries[MBV01_KEY]["matbench_steels"]}})
            self.assertEqual(render.call_args.args[0].layout.yaxis.type, "log")

    def test_scaled_chart_keeps_both_auc_boundaries_without_fake_epsilon(self):
        graph = {MBV01_KEY: {self.task: {"AUC zero": 0.0, "AUC half": 0.5, "AUC one": 1.0}}}
        with patch.object(docs.pio, "to_html", return_value="chart") as render:
            docs.generate_scaled_errors_graph(graph, output_fname="not-written")
            fig = render.call_args.args[0]
        self.assertEqual(fig.layout.yaxis.type, "linear")
        values = {trace.name: list(trace.y) for trace in fig.data if trace.name != "Best algorithms"}
        self.assertEqual(values, {"AUC zero": [2.0], "AUC half": [1.0], "AUC one": [0.0]})
        self.assertIn("zero scaled error", fig.layout.title.text)
        graph = {MBV01_KEY: {self.task: {"AUC zero": 0.0, "AUC half": 0.5}}}
        with patch.object(docs.pio, "to_html", return_value="chart") as render:
            docs.generate_scaled_errors_graph(graph, output_fname="not-written")
            self.assertEqual(render.call_args.args[0].layout.yaxis.type, "log")

    def test_tied_scores_choose_same_model_independent_of_input_order(self):
        for regression in (False, True):
            name = "matbench_steels" if regression else self.task
            for reverse in (False, True):
                with self.subTest(regression=regression, reverse=reverse):
                    items = [("z", packet("Z model", regression=regression)),
                             ("a", packet("A model", regression=regression))]
                    if reverse:
                        items.reverse()
                    gp, entries, gp_graph, _, _ = docs.organize_task_data(dict(items))
                    self.assertEqual(gp[MBV01_KEY][name]["algorithm"], "A model")
                    # Test the table's own tie policy, independent of organizer order.
                    entries[MBV01_KEY][name].reverse()
                    with tempfile.TemporaryDirectory() as tmp, patch.object(docs, "PER_TASK_DIR", tmp), patch.object(docs.pio, "to_html", return_value="chart") as render:
                        docs.generate_per_task_leaderboards({MBV01_KEY: {name: entries[MBV01_KEY][name]}})
                        text = (Path(tmp) / f"{MBV01_KEY}_{name}.md").read_text().split("<details>", 1)[0]
                        self.assertLess(text.index("| [A model]"), text.index("| [Z model]"))
                        docs.generate_scaled_errors_graph(gp_graph, output_fname="not-written")
                        best = next(trace for trace in render.call_args.args[0].data if trace.name == "Best algorithms")
                        self.assertEqual(list(best.text), ["A model"])

    def test_empty_probability_rankings_render_without_promoting_hard_labels(self):
        gp, entries, gp_graph, all_graph, counts = docs.organize_task_data({"bool": self.bool_packet})
        self.assertIsNone(gp[MBV01_KEY][self.task]["algorithm"])
        self.assertEqual(gp_graph[MBV01_KEY][self.task], {})
        with tempfile.TemporaryDirectory() as tmp, patch.object(docs, "PER_TASK_DIR", tmp), patch.object(docs, "DOCS_DIR", tmp), patch.object(docs, "STATIC_DOCS_DIR", tmp):
            docs.generate_per_task_leaderboards({MBV01_KEY: {self.task: entries[MBV01_KEY][self.task]}})
            text = (Path(tmp) / f"{MBV01_KEY}_{self.task}.md").read_text()
            current, history = text.split("<details>", 1)
            self.assertIn("No eligible probability submissions", current)
            self.assertNotIn("Boolean model", current)
            self.assertIn("Boolean model", history)
            docs.generate_general_purpose_leaderboard_and_plot(gp, counts)
            self.assertIn("No eligible general-purpose probability submissions", (Path(tmp) / "index.md").read_text())
            graph = docs.generate_scaled_errors_graph(gp_graph, output_fname="not-written")
            self.assertIn("No eligible scores", graph)


if __name__ == "__main__":
    unittest.main()

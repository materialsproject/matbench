import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
spec = importlib.util.spec_from_file_location(
    "glass_runner_test", HERE / "src/run_matbench_glass.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CompositionProtocolTest(unittest.TestCase):
    def test_auc_uses_probabilities_below_the_label_threshold(self):
        # All labels would be False at 0.5, but probability ordering is perfect.
        scores = runner.matbench_score_array(
            [False, False, True, True], [0.1, 0.2, 0.3, 0.4]
        )
        self.assertEqual(scores["rocauc"], 1.0)
        self.assertEqual(scores["balanced_accuracy"], 0.5)

    def test_outer_target_changes_do_not_change_fitted_predictions_or_selection(
        self,
    ):
        import numpy as np
        import pandas as pd
        from matbench.task import MatbenchTask
        import protocol

        self.assertTrue(
            callable(getattr(protocol, "record_fold", None)),
            "a fold must train/select before official recording sees test labels",
        )
        original = MatbenchTask("matbench_glass", autoload=True)
        changed = MatbenchTask("matbench_glass", autoload=False)
        changed.df = original.df.copy(deep=True)
        test_ids = original.validation.fold_0.test
        changed.df.loc[test_ids, original.metadata.target] = ~changed.df.loc[
            test_ids, original.metadata.target
        ]
        # Label-free real composition descriptors; no synthetic model or scoring stub.
        matrix = pd.DataFrame(
            [protocol.feature_vector(f) for f in original.df.composition],
            index=original.df.index,
        )
        grid = (
            {"max_iter": 5, "max_leaf_nodes": 3, "learning_rate": 0.1},
            {"max_iter": 8, "max_leaf_nodes": 5, "learning_rate": 0.1},
        )
        protocol.record_fold(original, matrix, 0, True, threads=1, grid=grid)
        protocol.record_fold(changed, matrix, 0, True, threads=1, grid=grid)
        self.assertEqual(
            dict(original.results.fold_0.data), dict(changed.results.fold_0.data)
        )
        self.assertEqual(
            dict(original.results.fold_0.parameters),
            dict(changed.results.fold_0.parameters),
        )
        self.assertNotEqual(
            original.results.fold_0.scores.rocauc,
            changed.results.fold_0.scores.rocauc,
        )


if __name__ == "__main__":
    unittest.main()

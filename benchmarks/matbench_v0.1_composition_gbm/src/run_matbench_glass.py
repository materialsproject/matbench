"""Compatibility entry point; all scoring uses the current official Matbench API."""

from matbench.data_ops import score_array


def matbench_score_array(true_array, pred_array):
    return score_array(list(true_array), list(pred_array), "classification")


if __name__ == "__main__":
    import runpy
    from pathlib import Path

    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "reproduce.py"),
        run_name="__main__",
    )

import os
import json
import logging

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
    pass


def generate_info_page(mb: MatbenchBenchmark, info: dict):

    is_complete = mb.is_complete

    algo_name = info["algorithm"]
    algo_desc = info["algorithm_long"]


    header = "# Data page for "





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
            results = MatbenchBenchmark.from_file(results_path)

            info = loadfn(info_path)

            name = info["algorithm"]
            all_data[name] = {"results": results, "info": info}


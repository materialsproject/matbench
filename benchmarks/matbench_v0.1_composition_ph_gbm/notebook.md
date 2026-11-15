# CompPHGBM: corrected standalone reproduction

This folder submits `matbench_log_gvrh` and `matbench_log_kvrh`, using official Matbench v0.1 IDs and folds.
It replaces the corresponding part of [PR #363](https://github.com/materialsproject/matbench/pull/363).

## Protocol and historical exposure

The input is the official structure only. Its reduced formula gives 156 composition features. The original deterministic PH featurizer gives 51 features: finite H0/H1 Vietoris–Rips persistence statistics and histograms plus unit-cell atom count, volume, density, point-cloud size and expansion-fallback flag. The combined 207-dimensional track is fixed. Neither a composition-only nor a PH-only outer score can change the submitted track.

Use periodic neighbors within 6 Å, sorted/deduplicated coordinates rounded to 8 decimals, filtration threshold 6 Å, H0/H1 and 8 histogram bins. If the coverage cloud exceeds 1000 points, use a deterministic 2×2×2 supercell; if it still exceeds 1000, retain the nearest points to the central-cell centroid. Both fallback and cap counts are measured in validation.json. The five geometry/density companions remain part of the descriptor; no claim isolates a benefit due solely to topology.

For every outer fold, use only `get_train_and_val_data()` to select the four
HistGradientBoosting configurations: `max_iter={300,600}` ×
`max_leaf_nodes={31,63}`, `learning_rate=0.1`, seed 42. The inner split is
90/10, seed 42, stratified for glass. Use probability ROC-AUC for classification
and MAE for regression. Refit the selected configuration on the complete
training fold, obtain test inputs with `include_target=False`, then call
`task.record()` once. No test targets or test scores enter fitting or selection.

The earlier work inspected outer scores for alternative feature/model tracks.
There is no recovered evidence proving the feature/model choice predated those
scores. This corrective rerun fixes the declared track before refitting, but
**does not make previously seen folds an unseen holdout**. Those earlier track
comparisons are exploratory; no best-of-outer-fold-track selection or leaderboard
rank comparison is performed by this runner.

## Install and execute

Use **CPython 3.12.10** in a fresh environment. All tested dependencies are pinned
in `requirements.txt`, including pymatgen and pyarrow. The runner now uses NPZ
rather than the original Parquet cache. Matbench's version string is still 0.6,
so its Git revision is pinned explicitly to `4b7d0714849a917814115be380a64e7ace94be55`; installing the older
PyPI `matbench==0.6` is not equivalent. The runner checks the official scorer and
fold-file hashes before training.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --work-dir run_outputs --threads 1 --workers 4
python reproduce.py --verify-only
```

`reproduce.py` executes data loading, deterministic features, training, official
recording/scoring and serialization. The final command reloads the committed
artifact with `MatbenchBenchmark.from_file()` and independently recomputes every
stored score from the official dataset and predictions. `--output PATH` can be
used to keep a second generated result for comparison. Compare per-fold
predictions, scores and selected parameters; timestamps, wall times and cache-hit
metadata can differ, so byte-identical gzip files are not claimed.

The command first writes `run_outputs/matbench_log_gvrh.json.gz` and
`run_outputs/matbench_log_kvrh.json.gz`, then merges the actual task objects
through the official API. To regenerate only the merged artifact:

```bash
python reproduce.py --work-dir run_outputs --merge-only
```

The merge rejects missing, duplicate, unexpected, invalid or incomplete task
runs. It retains each task's parameters, predictions, scores and metadata and
validates the merged `results.json.gz` with `MatbenchBenchmark.from_file()`.

## Data, cache and provenance

The pinned official Matbench/Matminer loaders obtain the data and check the
registry's SHA256 hashes. The official loader assigns mbids; this submission
does not reconstruct row IDs. The bundled validation file supplies all folds;
no live `main` fold download, leaderboard HTTP request, custom split or outside
training data is used.

Feature caching uses inputs only, source hashes, numerical-library versions and
fixed featurizer settings. The full ordered-input/code key, matrix shape, finite values and cache
checksums are verified on reuse. Rows are bound to the requesting task's official
IDs, so the different GVRH/KVRH ID prefixes do not block valid cache reuse. The structure cache can be shared between the
two regression tasks only if their complete ordered structures match; targets
never enter the cache key or features. Generated caches stay under `run_outputs`
and are not committed.

Each run writes a per-task `_run.json` with dataset/source/SDK hashes, selected
inner settings, measured fold and feature durations, CPU/OS, Python/dependency
versions, thread/worker settings, and output hash. The submission's checked
summary is `validation.json`; historical compute details that were not recovered
are not invented. Timings are measured wall times for this run, not portable
performance guarantees.

## Results and checks

Regenerated five-fold mean MAE:

| Task | MAE | Fold standard deviation | Measured task wall time |
|---|---:|---:|---:|
| `matbench_log_gvrh` | 0.08104860 | 0.00114610 | 468.25 s |
| `matbench_log_kvrh` | 0.06396968 | 0.00188536 | 343.38 s |

Apple M1, 16 GiB RAM, macOS arm64, one model thread. PH feature generation
used four workers and took 57.75 s within the first task; the second task reused
the verified identical ordered-structure cache. Both tasks report zero expansion
fallbacks and zero hard caps among 10,987 structures. No topology-only or
leaderboard-rank improvement is claimed.

See `validation.json` for complete numerical results and hardware/timing
records. Official serialization supplies all required metrics, including MAPE
for regressions. The scores are computed by the pinned official scorer, not
by the original hand-written regression scorer that omitted MAPE.

Run the bundled `test_*.py` with `python -m pytest`. The regression tests check
probability scoring, independence of predictions/configuration from changed
outer labels, and complete/duplicate-safe task merging as applicable. The
upstream Matbench submission validator and core test suite are also run before
submission. Upstream dependency deprecation warnings are documented separately
from test failures.

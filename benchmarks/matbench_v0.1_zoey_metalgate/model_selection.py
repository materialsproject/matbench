"""
How the Zoey-MetalGate configuration was chosen, WITHOUT touching test data.

This file documents and reproduces the selection procedure that produced the
blend in run.py. It never loads test targets: every score computed here comes
from internal cross-validation inside the official TRAINING folds.

PROCEDURE
---------
Stage 1 "screen": each candidate configuration is scored by internal 5-fold CV
on the training split of official fold 0 only. This is a cheap ranking pass;
103 configurations were screened across two campaigns (XGBoost, LightGBM and
CatBoost hyperparameters, three feature-set sizes, plain regressors, seed-bagged
regressors, and metal-gated two-stage models with soft and hard gates).

Stage 2 "confirm": the best configurations are re-scored by internal 5-fold CV
inside EACH of the five official training folds (25 model fits per config), and
their out-of-fold predictions are saved. A family-diversity cap (at most four
configurations per model family) keeps the candidate pool from collapsing onto
near-identical models.

Stage 3 "blend": greedy forward selection with replacement over the confirmed
out-of-fold predictions, minimising mean out-of-fold MAE across the five
training folds. This chose three metal-gated models with weights 0.5/0.25/0.25.

Only after all of the above was the finalist frozen to a file and evaluated
once on the official test splits by run.py.

VALIDATION SCORES THAT DROVE THE CHOICE (internal CV, training folds only)
-------------------------------------------------------------------------
    plain LightGBM, 493 features .................... 0.3465
    plain XGBoost, 493 features ..................... 0.3596
    metal-gated, hard gate, LightGBM regressor ...... 0.2965
    metal-gated, hard gate, XGBoost regressor ....... 0.2963
    metal-gated, soft gate, XGBoost regressor ....... 0.2915
    three-member blend (submitted) .................. 0.2903

For reference, the corresponding official test score of the submitted blend is
0.2775, and of the plain LightGBM 0.3322. The gap between validation and test
is consistent with training on the full fold rather than 4/5 of it, an effect
that was measured on an earlier model before this campaign ran.

WHY THE GATE HELPS, ARITHMETICALLY
----------------------------------
On out-of-fold predictions the classifier agrees with the true metal/nonmetal
label about 92% of the time. Metals (53% of rows) then carry roughly 0.15 eV of
residual error and non-metals (47%) roughly 0.45 eV, and
0.53 * 0.15 + 0.47 * 0.45 is about 0.29, which is the validation score above.
The improvement is concentrated in the metal population and in wide-gap
insulators; the soft gate slightly HURTS compounds with true gaps in the
0 to 1 eV range, which is the honest cost of the mechanism.
"""

from __future__ import annotations

import numpy as np
from matbench.bench import MatbenchBenchmark
from sklearn.model_selection import KFold

from run import BLEND, build_features

CV_SEED = 18012019
N_INTERNAL_FOLDS = 5


def internal_cv_score():
    """Reproduce the validation score of the submitted blend. No test data."""
    mb = MatbenchBenchmark(autoload=False, subset=["matbench_expt_gap"])
    task = mb.tasks_map["matbench_expt_gap"]
    task.load()
    features = build_features(task.df["composition"])

    fold_maes = []
    for fold in task.folds:
        X_train, y_train = task.get_train_and_val_data(fold)
        X = features.loc[X_train.index].values
        y = np.asarray(y_train.values, dtype=float)

        oof = np.zeros(len(y))
        kf = KFold(n_splits=N_INTERNAL_FOLDS, shuffle=True, random_state=CV_SEED)
        for inner_train, inner_val in kf.split(X):
            pred = np.zeros(len(inner_val))
            for weight, make_model in BLEND:
                model = make_model().fit(X[inner_train], y[inner_train])
                pred += weight * model.predict(X[inner_val])
            oof[inner_val] = pred

        mae = float(np.mean(np.abs(oof - y)))
        fold_maes.append(mae)
        print(f"official training fold {fold}: internal-CV MAE {mae:.4f}")

    print(f"\nmean internal-CV MAE: {np.mean(fold_maes):.4f} "
          f"(+/- {np.std(fold_maes):.4f})")


if __name__ == "__main__":
    internal_cv_score()

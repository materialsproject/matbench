"""Amendment-4 expert pool (ANALYSIS_PLAN.md Section 14): the frozen Amendment-2 pool (src/experts2.py, unchanged)
with the median imputation of the `extra` and CBFV Oliynyk blocks moved INSIDE every fit.

Amendments 2/3 filled the missing cells of those two blocks once, with column medians over all rows of a task
(training and test rows together). Here the cells arrive as NaN (src/features4.py) and each expert fit computes the
column medians from the rows it is fitted on, then applies them to the rows it predicts:
  - final refit on the official training fold T_k  -> medians of T_k, applied to T_k and to the test rows of fold k;
  - every inner-CV fit (cross-fitting, Optuna trials) -> medians of that inner-training part, applied to it and to
    its inner-validation part.
A column that is NaN in every fit row is filled with 0.0 (a constant: it carries no information for that fit).
Models, hyperparameters, seeds, feature groups and everything else are the Amendment-2 code.
"""
from __future__ import annotations
from .experts2 import Expert2, make_experts2, group_cols
from .features4 import IMPUTED_PREFIX


class Expert4(Expert2):
    med = None

    def clone(self):
        e = Expert4(self.name, self.group, self.factory, self.task_type, self.tunable)
        e.params = dict(self.params)
        return e

    def _X(self, X):
        Xs = X[group_cols(X, self.group)]
        if self.med is not None and len(self.med):
            Xs = Xs.fillna(self.med)           # fills only the columns in self.med, with the FIT rows' medians
        return Xs

    def fit(self, X, y):
        cols = [c for c in group_cols(X, self.group) if c.startswith(IMPUTED_PREFIX)]
        self.med = X[cols].median().fillna(0.0)        # medians of the rows this fit sees, nothing else
        return super().fit(X, y)


def make_experts4(task_type, n_train, seed=0, cfg=None):
    out = []
    for e in make_experts2(task_type, n_train, seed=seed, cfg=cfg):
        assert e.group != "comp"
        x = Expert4(e.name, e.group, e.factory, e.task_type, e.tunable)
        x.params = dict(e.params)
        out.append(x)
    return out

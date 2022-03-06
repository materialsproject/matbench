import pandas as pd
from ax import Data
from ax.storage.metric_registry import register_metric
from ax.core import Metric
from utils.parameterization import crabnet_mae

# %% CrabNetMetric
class CrabNetMetric(Metric):
    def __init__(self, name, train_val_df, n_splits=5):
        self.train_val_df = train_val_df
        self.n_splits = n_splits
        super().__init__(name=name)

    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters

            # TODO: add timing info as optional parameter and as outcome metric
            # TODO: maybe add interval score calculation as outcome metric
            mean = crabnet_mae(
                params, self.train_val_df, n_splits=self.n_splits
            )

            records.append(
                {
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "trial_index": trial.index,
                    "mean": mean,
                    "sem": None,
                }
            )
        return Data(df=pd.DataFrame.from_records(records))


register_metric(metric_cls=CrabNetMetric)

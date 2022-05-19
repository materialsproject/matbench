import pprint
import gc
import numpy as np
from torch.cuda import empty_cache
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from crabnet.train_crabnet import get_model


def correct_parameterization(parameterization, verbose=False):
    if verbose:
        pprint.pprint(parameterization)

    parameterization["out_hidden"] = [
        parameterization.get("out_hidden4") * 8,
        parameterization.get("out_hidden4") * 4,
        parameterization.get("out_hidden4") * 2,
        parameterization.get("out_hidden4"),
    ]
    parameterization.pop("out_hidden4")

    parameterization["betas"] = (
        parameterization.get("betas1"),
        parameterization.get("betas2"),
    )
    parameterization.pop("betas1")
    parameterization.pop("betas2")

    d_model = parameterization["d_model"]

    # make heads even (unless it's 1) (because d_model must be even)
    heads = parameterization["heads"]
    if np.mod(heads, 2) != 0:
        heads = heads + 1
    parameterization["heads"] = heads

    # NOTE: d_model must be divisible by heads
    d_model = parameterization["heads"] * round(d_model / parameterization["heads"])

    parameterization["d_model"] = d_model

    parameterization["pos_scaler_log"] = (
        1 - parameterization["emb_scaler"] - parameterization["pos_scaler"]
    )

    parameterization["epochs"] = parameterization["epochs_step"] * 4

    return parameterization


def crabnet_mae(parameterization, train_val_df, n_splits=5, kf=None, verbose=False):
    """Compute the mean absolute error of a CrabNet model.
    
    Assumes that `train_df` and `val_df` are predefined.

    Parameters
    ----------
    parameterization : dict
        Dictionary of the parameters passed to `get_model()` after some slight
        modification. 

    Returns
    -------
    results: dict
        Dictionary of `{"rmse": rmse}` where `rmse` is the root-mean-square error of the
        CrabNet model.
    """
    parameterization = correct_parameterization(parameterization, verbose=verbose)

    if kf is None:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=18012019)

    mae = 0.0
    for train_index, val_index in kf.split(train_val_df):
        train_df, val_df = (
            train_val_df.loc[train_index],
            train_val_df.loc[val_index],
        )
        crabnet_model = get_model(
            mat_prop="expt_gap",
            train_df=train_df,
            learningcurve=False,
            force_cpu=False,
            verbose=verbose,
            **parameterization
        )
        val_true, val_pred, val_formulas, val_sigma = crabnet_model.predict(val_df)
        # rmse = mean_squared_error(val_true, val_pred, squared=False)
        val_pred = np.nan_to_num(val_pred)
        mae = mae + mean_absolute_error(val_true, val_pred)

        # deallocate CUDA memory https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/28
        del crabnet_model
        gc.collect()
        empty_cache()
    mae = mae / n_splits
    return mae

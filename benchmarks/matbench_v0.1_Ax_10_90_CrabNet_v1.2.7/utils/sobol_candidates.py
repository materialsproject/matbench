# import string
import numpy as np
import pandas as pd

from itertools import combinations, product, chain

from scipy.stats.qmc import Sobol, discrepancy
from sklearn.preprocessing import normalize
from utils.fractional import fractional_encode

# n_slots = 5
# n_vars = 13
# m = 15
# use_sobol = False

# unique_components = list(string.ascii_lowercase)[0:n_vars]
# unique_components = ["a", "b", "c", "d", "e", "f", "g"]


def sobol_candidates(
    unique_components, n_slots=5, m=15, scale=1.0, use_sobol=True, verbose=False
):

    n_unique = len(unique_components)

    num_pts = 2 ** m
    if verbose:
        print("number of points: ", num_pts)

    if use_sobol:
        sampler = Sobol(n_slots)
        sampler2 = Sobol(n_slots)
        component_ids = sampler.random_base2(m)
        compositions = sampler2.random_base2(m)

    else:
        component_ids = np.random.rand(num_pts, n_slots)
        compositions = np.random.rand(num_pts, n_slots)

    if verbose:
        print("component_ids discrepancy:", discrepancy(component_ids))
        print("compositions discrepancy:", discrepancy(compositions))

    component_ids = n_unique * component_ids
    component_ids = np.floor(component_ids)

    compositions = scale * normalize(compositions, norm="l1")

    # https://stackoverflow.com/a/33529918/13697228
    transdict = dict(zip(range(n_unique), unique_components))
    keys = list(transdict.keys())
    values = list(transdict.values())
    sort_idx = np.argsort(keys)
    idx = np.searchsorted(keys, component_ids, sorter=sort_idx)
    components = np.asarray(values)[sort_idx][idx]

    X_train = fractional_encode(
        components, compositions, unique_components=unique_components
    )

    if verbose:
        print("X_train discrepancy:", discrepancy(X_train))

    return X_train


def nonsparse_compositions(
    n_slots=5, comb_m=5, scale=1.0, use_sobol=True, ncombs=1, seed=None, verbose=False,
) -> list:
    npts_per_comb = 2 ** comb_m  # e.g. 2**10 = 1024
    compositions = []
    for _ in range(ncombs):
        if seed is not None:
            seed = seed + 1
        if use_sobol:
            sampler = Sobol(n_slots, seed=seed)
            composition = sampler.random_base2(comb_m)
        else:
            if seed is not None:
                np.random.rand(seed)
            # can be sped up by using np.random.rand(npts_per_comb * ncombs)
            composition = np.random.rand(npts_per_comb)
        composition = scale * normalize(composition, norm="l1")
        compositions.append(composition.tolist())
    if verbose:
        print("compositions[0] discrepancy:", discrepancy(compositions[0]))
    if ncombs == 1:
        compositions = compositions[0]
    return compositions


def nchoosek_sobol(
    unique_components,
    n_slots=5,
    comb_m=5,
    scale=1.0,
    use_sobol=True,
    fixed_compositions=True,
    verbose=False,
    seed=None,
    return_ncombs=False,
):
    component_combs = list(combinations(unique_components, n_slots))
    # 13 choose 5 ==> 1287 combs # https://www.hackmath.net/en/calculator
    ncombs = len(component_combs)
    npts_per_comb = 2 ** comb_m  # 2**10 = 1024
    if verbose:
        print("number of nchoosek combinations: ", ncombs)
        print("number of points per combination (2**comb_m): ", npts_per_comb)
        print("total points: ", ncombs * npts_per_comb)  # default, ~1.3e6

    if fixed_compositions:
        ncombs = 1
    compositions = nonsparse_compositions(
        n_slots=n_slots,
        comb_m=comb_m,
        scale=scale,
        use_sobol=use_sobol,
        ncombs=ncombs,
        seed=seed,
        verbose=verbose,
    )

    if fixed_compositions:
        all_combs = list(product(component_combs, compositions))
    else:
        all_combs = [
            list(product([comb], composition))
            for comb, composition in zip(component_combs, compositions)
        ]
        # flatten https://stackabuse.com/python-how-to-flatten-list-of-lists/
        all_combs = list(chain(*all_combs))
    parameters = [dict(list(zip(*comb))) for comb in all_combs]
    df = pd.DataFrame(parameters).fillna(0)
    # rearrange columns https://stackoverflow.com/a/13148611/13697228
    df = df[unique_components]
    # shuffle DataFrame rows https://stackoverflow.com/a/34879805/13697228
    X_train = df.sample(frac=1, random_state=seed)

    if verbose:
        print("X_train discrepancy:", discrepancy(X_train))

    if not return_ncombs:
        return X_train
    else:
        return X_train, ncombs

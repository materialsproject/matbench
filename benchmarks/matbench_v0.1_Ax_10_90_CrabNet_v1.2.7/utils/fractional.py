"""Helper functions related to fractional encoding and decoding."""
from collections.abc import Iterable
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def flatten(list_of_lists):
    """Flatten ragged list of lists.

    Modified from: https://stackoverflow.com/a/2158532/13697228

    Parameters
    ----------
    l : list of lists
        Ragged "array" (list of lists) to be flattened.

    Yields
    -------
    flattened_list
        Generator for flattened list (i.e. call list(flattened_list) to convert to list)
    """
    for el in list_of_lists:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def fractional_encode(
    components, compositions, unique_components=None, drop_last=False
):
    """Fractionally encode components and compositions similar to one-hot encoding.

    In one-hot encoding, components are assigned a "1" if it exists for a particular
    compound, and a "0" if it does not. However, this ignores the case where the
    composition (i.e. the fractional prevalence) of each component is known. For
    example, NiAl is 50% Ni and 50% Al. This function computes the fractional components
    (albeit manually using for loops) where instead of a "1" or a "0", the corresponding
    fractional prevalence is assigned (e.g. 0.2, 0.5643, etc.).

    Modified from: https://stackoverflow.com/a/70224601/13697228

    Parameters
    ----------
    components : (list of lists or ragged array) of strings or numbers
        The components that make up the compound for each compound. If strings, then
        each string corresponds to a category. If numbers, then each number must
        uniquely describe a particular category.
    compositions : (list of lists or ragged array) of floats
        The compositions of each component that makes up the compound for each compound.
    unique_components : list of strings or numbers
        The unique components available for the algorithm to choose from. If None, then `unique_components` is created by taking the unique values
        across all `components`. By default None
    drop_last : bool, optional
        Whether to drop the last component. This can be useful since compositions are
        constrained to sum to one, and therefore there are `n_components - 1` degrees of freedom, by default False

    Returns
    -------
    X_train : DataFrame
        Fractionally encoded matrix with column names given by `unique_components`.

    Raises
    ------
    ValueError
        Components and compositions should have the same shape.

    See also
    --------
    "Convert jagged array to Pandas dataframe"
    https://stackoverflow.com/a/63496196/13697228

    Example(s)
    ---------
    components = np.array(
        [
            ["filler_A", "filler_B", "resin_C"],
            ["filler_A", "resin_B"],
            ["filler_A", "filler_B", "resin_B"],
            ["filler_A", "resin_B", "resin_C"],
            ["filler_B", "resin_A", "resin_B"],
            ["filler_A", "resin_A"],
            ["filler_B", "resin_A", "resin_B"],
        ],
        dtype=object,
    )
    compositions = np.array(
        [
            [0.4, 0.4, 0.2],
            [0.5, 0.5],
            [0.5, 0.3, 0.2],
            [0.5, 0.5, 0.0],
            [0.6, 0.4, 0.0],
            [0.6, 0.4],
            [0.6, 0.2, 0.2],
        ],
        dtype=object,
    )
    unique_components = ["filler_A", "filler_B", "resin_A", "resin_B", "resin_C"]
    X_train = fractional_encode(components, compositions, unique_components=None)
    print(X_train)
    >         filler_A  filler_B  resin_A  resin_B  resin_C
    > 0       0.4       0.4      0.0      0.0      0.2
    > 1       0.5       0.0      0.0      0.5      0.0
    > 2       0.5       0.3      0.0      0.2      0.0
    > 3       0.5       0.0      0.0      0.5      0.0
    > 4       0.0       0.6      0.4      0.0      0.0
    > 5       0.6       0.0      0.4      0.0      0.0
    > 6       0.0       0.6      0.2      0.2      0.0
    """
    n_compounds = len(components)
    if unique_components is None:
        unique_components = sorted(list(set(sum(list(components), []))))

    # initialize
    X_train = np.zeros((len(components), len(unique_components)))

    for i in range(n_compounds):
        for component, composition in zip(components[i], compositions[i]):
            j = unique_components.index(component)
            X_train[i, j] = composition

    if drop_last:
        # remove last column: https://stackoverflow.com/a/6710726/13697228
        X_train = np.delete(X_train, -1, axis=1)
    X_train = pd.DataFrame(data=X_train, columns=unique_components)
    return X_train


def fractional_decode(X_train, sort=True, reverse=True):
    """Fractionally decode components and compositions similar to one-hot encoding.

    In one-hot encoding, components are assigned a "1" if it exists for a particular
    compound, and a "0" if it does not. However, this ignores the case where the
    composition (i.e. the fractional prevalence) of each component is known. For
    example, NiAl is 50% Ni and 50% Al. This function decodes the fractional encoding
    where instead of "1" or a "0", the corresponding fractional prevalence is used (e.g. 0.2, 0.5643, etc.).

    Parameters
    ----------
    X_train : DataFrame
        Fractionally encoded matrix (similar to a one-hot encoded matrix).
    sort : bool, optional
        Whether to sort components/compositions by the composition values. If ascending
        is True, then sort in ascending order. By default, True.
    reverse : bool, optional
        Only applies if sort is True. Whether to sort in ascending order. By default, True.

    Returns
    -------
    components : list of lists of strings or numbers
        The components that make up the compound for each compound. If strings, then
        each string corresponds to a category. If numbers, then each number must
        uniquely describe a particular category.
    compositions : list of lists of floats
        The compositions of each component that makes up the compound for each compound.

    Raises
    ------
    ValueError
        Components and compositions should have the same shape.

    Example(s)
    ---------
    X_train = np.array(
       [[0.4, 0.4, 0. , 0. , 0.2],
       [0.5, 0. , 0. , 0.5, 0. ],
       [0.5, 0.3, 0. , 0.2, 0. ],
       [0.5, 0. , 0. , 0.5, 0. ],
       [0. , 0.6, 0.4, 0. , 0. ],
       [0.6, 0. , 0.4, 0. , 0. ],
       [0. , 0.6, 0.2, 0.2, 0. ]])
    unique_components = ["filler_A", "filler_B", "resin_A", "resin_B", "resin_C"]
    X_train = pd.DataFrame(data=X_train, columns=unique_components)

    components, compositions = fractional_decode(X_train)

    print(components)
    > [('filler_A', 'filler_B', 'resin_C'), ('filler_A', 'resin_B'), ('filler_A',
    > 'filler_B', 'resin_B'), ('filler_A', 'resin_B'), ('filler_B', 'resin_A'),
    > ('filler_A', 'resin_A'), ('filler_B', 'resin_A', 'resin_B')]

    print(compositions)
    > [(0.4, 0.4, 0.2), (0.5, 0.5), (0.5, 0.3, 0.2), (0.5, 0.5), (0.6, 0.4), (0.6, 0.4)> , (0.6, 0.2, 0.2)]
    """
    # lengths, unique components, and sparse matrix attributes
    unique_components = X_train.columns
    n_unique = len(unique_components)
    sparse_mat = coo_matrix(X_train.values)
    row_ids, col_ids = sparse_mat.row, sparse_mat.col
    idx_pairs = list(zip(row_ids, col_ids))
    comps = sparse_mat.data

    # lookup dictionaries to replace col_ids with components
    component_lookup = {
        component_idx: unique_component
        for (component_idx, unique_component) in zip(range(n_unique), unique_components)
    }

    # lookup dictionaries to replace idx_pairs with compositions
    composition_lookup = {idx_pair: comp for (idx_pair, comp) in zip(idx_pairs, comps)}

    # contains placeholder col_ids and idx_pairs which will get replaced by components
    # and compositions, respectively
    tmp_df = pd.DataFrame(
        data=[(idx_pair[1], idx_pair) for idx_pair in idx_pairs],
        columns=["component", "composition"],
    )

    # NOTE: component_lookup should be mapped before composition_lookup
    tmp_df.component = tmp_df.component.map(component_lookup)
    tmp_df.composition = tmp_df.composition.map(composition_lookup)

    # add a row_id column to use for grouping into ragged entries
    cat_df = pd.concat([pd.DataFrame(row_ids, columns=["row_id"]), tmp_df], axis=1)

    # combine components and compositions compound-wise
    df = (
        cat_df.reset_index()
        .groupby(by="row_id")
        .agg({"component": lambda x: tuple(x), "composition": lambda x: tuple(x)})
    )

    # extract and convert to ragged lists
    components, compositions = [df[key] for key in ["component", "composition"]]
    components = list(components)
    compositions = list(compositions)

    if sort:
        components, compositions = sort_data_by_compositions(
            components, compositions, reverse=reverse
        )

    return components, compositions


def append_last_component(parameters, unique_components, max=1.0):
    """Append the last component which brings composition to unity.

    Parameters
    ----------
    parameters : dict (String -> float)
        Ax parameters (all except last of `unique_components`).
    unique_components : list of strings
        Unique components which contains all of parameter names and the `last_component`
        name.
    """
    x = np.array([parameters.get(component) for component in unique_components[:-1]])
    last_component = max - sum(x)
    x = np.append(x, last_component)
    next_experiment = parameters
    next_experiment[unique_components[-1]] = last_component
    return next_experiment


def count_nonzero_components(df, tol=1e-6):
    """Count how many components are above some threshold.

    Parameters
    ----------
    parameters : dict
        Compositional data with `component`: `composition` key-value pairs.
    tol : float
        positive tolerance below which components are considered to be non-existent.

    Returns
    -------
    n_components : int or list of int
        Number of non-(close-to)-zero components.
    """
    df[df < tol] = 0.0
    n_components = np.count_nonzero(df, axis=1)
    return n_components


def sort_data_by_compositions(components, compositions, reverse=True):
    """Sort components and compositions by composition values.

    Parameters
    ----------
    components : list of list of str
        Can be ragged. The components that correspond to a given compound.
    compositions : list of list of str
        Can be ragged. Must be same "ragged shape" as components. The compositions that
        correspond to the components in a given compound.
    reverse : bool, optional
        Whether to sort in descending order, by default True

    Returns
    -------
    sorted_components : list of list of str
        Components sorted by the respective composition values. If reverse=True, then
        sorted in ascending order.
    sorted_compositions : list of list of str
        Sorted composition values. If reverse=True, then sorted in ascending order.
    """
    n_compounds = len(components)
    # initialize
    sorted_components = [None] * n_compounds
    sorted_compositions = [None] * n_compounds
    # loop through each compound
    for i, (X, Y) in enumerate(zip(components, compositions)):
        # sort list by second list: https://stackoverflow.com/a/6618543/13697228
        data = [(x, y) for y, x in sorted(zip(Y, X), reverse=reverse)]
        # transpose list of lists https://stackoverflow.com/a/6473724/13697228
        sorted_components[i], sorted_compositions[i] = list(map(list, zip(*data)))

    return sorted_components, sorted_compositions

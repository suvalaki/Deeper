import tensorflow as tf
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
import io
from typing import Union, Tuple
from collections import namedtuple

SplitCovariates = namedtuple(
    "SplitInputs",
    [
        "regression",
        "binary",
        "ordinal_groups_concat",
        "ordinal_groups",
        "categorical_groups_concat",
        "categorical_groups",
    ],
)


def chain_call(func, x, num, scalar_dict={}):

    if type(x) != list and type(x) != tuple:
        x = [x]

    iters = x[0].shape[0] // num

    result = []
    j = 0
    while j * num < x[0].shape[0]:
        result = result + [
            func(
                *(y[j * num : min((j + 1) * num, y.shape[0])] for y in x),
                **scalar_dict
            )
        ]
        j += 1

    num_dats = len(result[0])
    # pivot the resultsif
    if type(result[0]) in [tuple, list]:
        result_pivot = [
            np.concatenate([y[i] for y in result], 0) for i in range(num_dats)
        ]
    elif type(result[0]) == dict:
        result_pivot = {
            k: np.concatenate([y[k] for y in result], 0)
            for k in result[0].keys()
        }
    else:
        result_pivot = np.concatenate(result, axis=0)
    return result_pivot


def plot_latent(latent_vectors):

    pca = PCA(2)
    X_pca = pca.fit_transform(latent_vectors)
    df_latent = pd.DataFrame(
        {
            "x1": X_pca[:, 0],
            "x2": X_pca[:, 1],
        }
    )

    f, (ax1) = plt.subplots(1, 1, sharey=True, figsize=(10, 10))
    ax1.scatter(
        df_latent.x1,
        df_latent.x2,
    )
    ax1.set_title("Latent Space")

    return f


def split_groups(x, group_dims: Union[Tuple[int], np.array]):

    if type(group_dims) == np.array:
        assert len(group_dims.shape) == 1

    tot_dim = sum(group_dims)
    x_grouped = [
        x[:, sum(group_dims[:i]) : sum(group_dims[: i + 1])]
        for i in range(len(group_dims))
    ]
    return x_grouped


def split_inputs(
    x,
    reg_dim: int,
    bool_dim: int,
    ord_dim_tup: Tuple[int],
    cat_dim_tup: Tuple[int],
) -> SplitCovariates:
    x_reg = x[:, :reg_dim] if reg_dim > 0 else x[:, 0:0]
    x_bin = x[:, reg_dim : (reg_dim + bool_dim)] if bool_dim > 0 else x[:, 0:0]

    # categorical dimensions need to be further broken up according to the size
    # of the input groups
    cat_dim = sum(cat_dim_tup)
    ord_dim = sum(ord_dim_tup)
    x_ord = (
        x[:, -(ord_dim + cat_dim) : -(cat_dim)] if ord_dim > 0 else x[:, 0:0]
    )
    x_cat = x[:, -cat_dim:] if cat_dim > 0 else x[:, 0:0]
    x_ord_grouped = split_groups(x_ord, ord_dim_tup)
    x_cat_grouped = split_groups(x_cat, cat_dim_tup)

    return SplitCovariates(
        x_reg, x_bin, x_ord, x_ord_grouped, x_cat, x_cat_grouped
    )

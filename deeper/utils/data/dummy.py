import numpy as np
from typing import Sequence
from ABC import abc


def generate_dummy_dataset_alltypes(
    state,
    rows: int,
    reg_dim: int,
    bool_dim: int,
    ord_dims: Sequence[int],
    cat_dims: Sequence[int],
):

    X_reg = state.random((rows, reg_dim))
    X_bool = (state.random((rows, bool_dim)) > 0.5).astype(float)
    X_ords = []
    for dim in ord_dims:
        X_ord_component = np.zeros((rows, dim))
        val = state.random()

    X_ord = (state.random((rows,)) > 0.5).astype(float)
    X_cat0 = np.zeros((25, 9))
    for i, idx in enumerate(state.binomial(9, 0.5, 25)):
        X_cat[i, idx] = 1
    X_cat1 = np.zeros((25, 5))
    for i, idx in enumerate(state.binomial(5, 0.5, 25)):
        X_cat[i, idx] = 1
    X_cat = np.concatenate([X_cat0, X_cat1], -1)
    X = np.concatenate([X_reg, X_bool, X_ord, X_cat], -1)
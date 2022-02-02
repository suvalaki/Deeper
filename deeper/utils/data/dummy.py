import numpy as np
from typing import Sequence
from dataclasses import dataclass


@dataclass
class MultipleObjectiveInput:
    X: np.array
    X_reg: np.array
    X_bool: np.array
    X_ord: Sequence[np.array]
    X_cat: Sequence[np.array]


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

    X_ord_group = []
    for s in ord_dims:
        X_ord_tmp = np.zeros((rows, s))
        for i, idx in enumerate(state.binomial(s, 0.5, rows)):
            for j in range(idx):
                X_ord_tmp[i, j] = 1
        X_ord_group.append(X_ord_tmp)
    X_ord = np.concatenate(X_ord_group, axis=-1)

    X_cat_group = []
    for s in cat_dims:
        X_cat_tmp = np.zeros((rows, s))
        for i, idx in enumerate(state.binomial(s - 1, 0.5, rows)):
            X_cat_tmp[i, idx] = 1
        X_cat_group.append(X_cat_tmp)
    X_cat = np.concatenate(X_cat_group, axis=-1)

    X = np.concatenate([X_reg, X_bool, X_ord, X_cat], -1)

    return MultipleObjectiveInput(X, X_reg, X_bool, X_ord, X_cat)

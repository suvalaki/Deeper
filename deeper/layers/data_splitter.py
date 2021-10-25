from __future__ import annotations
import tensorflow as tf
import numpy as np
from deeper.utils.scope import Scope
from typing import Optional, Sequence, NamedTuple

tfk = tf.keras
Layer = tfk.layers.Layer
Model = tfk.Model

from dataclasses import dataclass, asdict


class SplitCovariates(NamedTuple):
    regression: tf.Tensor
    binary: tf.Tensor
    ordinal_groups_concat: tf.Tensor
    ordinal_groups: Sequence[tf.Tensor]
    categorical_groups_concat: tf.Tensor
    categorical_groups: Sequence[tf.Tensor]


def reduce_groups(fn, x_grouped: Sequence[tf.Tensor]):
    if len(x_grouped) <= 1:
        return fn(x_grouped[0])
    return tf.concat([fn(z) for z in x_grouped], -1)


def unpack_dimensions(
    reg_dim: int,
    bool_dim: int,
    ord_dim: Union[int, Sequence],
    cat_dim: Union[int, Sequence],
):
    ord_dim = ord_dim if isinstance(ord_dim, Sequence) else (ord_dim,)
    cat_dim = cat_dim if isinstance(cat_dim, Sequence) else (cat_dim,)
    tot_ord_dim = sum(ord_dim)
    tot_cat_dim = sum(cat_dim)
    tot_dim = reg_dim + bool_dim + tot_ord_dim + tot_cat_dim
    return (
        ord_dim,
        tot_ord_dim,
        cat_dim,
        tot_cat_dim,
        tot_dim,
    )


def split_groups(x, group_dims: Union[Tuple[int], np.array]):

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

    ord_dim = sum(ord_dim_tup)
    cat_dim = sum(cat_dim_tup)

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


@dataclass
class Config:
    dim_regression: Optional[int] = None
    dim_boolean: Optional[int] = None
    dim_ordinal: Optional[Union[int, Sequence[int]]] = None
    dim_categorical: Optional[Union[int, Sequence[int]]] = None


class DataSplitter(Layer):
    Config = Config

    def __init__(self, config: DataSplitter.Config, **kwargs):
        super().__init__(self, **kwargs)

        for k, v in asdict(config).items():
            setattr(self, k, v)

        (
            self.dim_ordinal,
            self.ordinal_dimension_tot,
            self.dim_categorical,
            self.categorical_dimension_tot,
            self.input_dim,
        ) = self.unpack_dimensions()

    def unpack_dimensions(self):
        return unpack_dimensions(
            self.dim_regression,
            self.dim_boolean,
            self.dim_ordinal,
            self.dim_categorical,
        )

    def call(self, x, training=False):
        return split_inputs(
            x,
            self.dim_regression,
            self.dim_boolean,
            self.dim_ordinal,
            self.dim_categorical,
        )
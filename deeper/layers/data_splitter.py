from __future__ import annotations
import tensorflow as tf
import numpy as np
from deeper.utils.scope import Scope
from typing import Optional, Sequence, NamedTuple, Tuple

tfk = tf.keras
Layer = tfk.layers.Layer
Model = tfk.Model

from dataclasses import dataclass, asdict
from deeper.utils.tf.experimental.extension_type import ExtensionTypeIterableMixin


class SplitCovariates(tf.experimental.ExtensionType, ExtensionTypeIterableMixin):
    regression: Optional[tf.Tensor] = None
    binary: Optional[tf.Tensor] = None
    ordinal_groups_concat: Optional[tf.Tensor] = None
    ordinal_groups: Optional[Tuple[tf.Tensor, ...]] = None
    categorical_groups_concat: Optional[tf.Tensor] = None
    categorical_groups: Optional[Tuple[tf.Tensor, ...]] = None


def reduce_groups(fn, x_grouped: Tuple[tf.Tensor, ...]):

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

    # x_grouped = [
    #     x[:, sum(group_dims[:i]) : sum(group_dims[: i + 1])] for i in range(len(group_dims))
    # ]

    x_grouped = [
        tf.gather(x, indices=tuple(range(sum(group_dims[:i]), sum(group_dims[: i + 1]))), axis=-1)
        if group_dims[i] > 0
        else tf.zeros(shape=tf.concat([tf.shape(x)[:-1], [0]], axis=-1))
        for i in range(len(group_dims))
    ]
    return x_grouped


@tf.function
def split_inputs(
    x,
    reg_dim: int,
    bool_dim: int,
    ord_dim_tup: Tuple[int],
    cat_dim_tup: Tuple[int],
) -> SplitCovariates:

    ord_dim = sum(ord_dim_tup)
    cat_dim = sum(cat_dim_tup)

    x_reg = (
        tf.gather(
            x,
            indices=tuple(range(0, reg_dim)),
            batch_dims=0,
            axis=-1,
        )
        if reg_dim > 0
        else tf.zeros(shape=tf.concat([tf.shape(x)[:-1], [0]], axis=-1))
    )
    x_bin = (
        tf.gather(
            x,
            indices=tuple(range(reg_dim, reg_dim + bool_dim)),
            batch_dims=0,
            axis=-1,
        )
        if bool_dim > 0
        else tf.zeros(shape=tf.concat([tf.shape(x)[:-1], [0]], axis=-1))
    )

    # categorical dimensions need to be further broken up according to the size
    # of the input groups
    cat_dim = sum(cat_dim_tup)
    ord_dim = sum(ord_dim_tup)

    x_ord = (
        tf.gather(
            x,
            indices=tuple(range(reg_dim + bool_dim, reg_dim + bool_dim + ord_dim)),
            batch_dims=0,
            axis=-1,
        )
        if ord_dim > 0 and (cat_dim > 0 or cat_dim == 0)
        else tf.zeros(shape=tf.concat([tf.shape(x)[:-1], [0]], axis=-1))
    )

    x_cat = (
        tf.gather(
            x,
            indices=tuple(
                range(reg_dim + bool_dim + ord_dim, reg_dim + bool_dim + ord_dim + cat_dim)
            ),
            batch_dims=0,
            axis=-1,
        )
        if cat_dim > 0
        else tf.zeros(shape=tf.concat([tf.shape(x)[:-1], [0]], axis=-1))
    )
    x_ord_grouped = split_groups(x_ord, ord_dim_tup)
    x_cat_grouped = split_groups(x_cat, cat_dim_tup)

    return SplitCovariates(x_reg, x_bin, x_ord, x_ord_grouped, x_cat, x_cat_grouped)


@dataclass
class Config:
    dim_regression: Optional[int] = None
    dim_boolean: Optional[int] = None
    dim_ordinal: Optional[Union[int, Sequence[int]]] = None
    dim_categorical: Optional[Union[int, Sequence[int]]] = None


class DataSplitter(Layer):
    Config = Config

    def __init__(self, config: DataSplitter.Config, **kwargs):
        super().__init__(**kwargs)

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

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional

from tensorflow.python.keras.metrics import (
    MeanMetricWrapper,
    categorical_accuracy,
)

from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops.losses import util as tf_losses_utils
from tensorflow.python.ops import math_ops

from deeper.layers.data_splitter import split_inputs, split_groups


def split_cat_true_pred_grouped(
    y_true,
    y_pred,
    reg_dim: int = 0,
    bool_dim: int = 0,
    ord_dim_tup: Tuple[int] = (0,),
    cat_dim_tup: Tuple[int] = (0,),
):
    yt_reg, yt_bin, yt_ord, yt_ord_grp, yt_cat, yt_cat_grp = split_inputs(
        y_true, reg_dim, bool_dim, ord_dim_tup, cat_dim_tup
    )
    yp_cat_grp = split_groups(y_pred["x_recon_cat_groups_concat"], cat_dim_tup)
    return yt_cat_grp, yp_cat_grp


def vae_categorical_dims_accuracy(
    y_true,
    y_pred,
    reg_dim: int = 0,
    bool_dim: int = 0,
    ord_dim_tup: Tuple[int] = (0,),
    cat_dim_tup: Tuple[int] = (0,),
    group_weights: Optional[int] = None,
):
    yt_cat_grp, yp_cat_grp = split_cat_true_pred_grouped(
        y_true,
        y_pred,
        reg_dim,
        bool_dim,
        ord_dim_tup,
        cat_dim_tup,
    )
    if group_weights is not None:
        grp_accs = tf.reduce_mean(
            tf.stack(
                [
                    wt * categorical_accuracy(ytg, ypg)
                    for (ytg, ypg, wt) in zip(
                        yt_cat_grp, yp_cat_grp, group_weights
                    )
                ]
            )
        )
    else:
        grp_accs = tf.reduce_mean(
            tf.stack(
                [
                    categorical_accuracy(ytg, ypg)
                    for (ytg, ypg) in zip(yt_cat_grp, yp_cat_grp)
                ]
            )
        )

    return grp_accs


class VaeCategoricalAvgAccuracy(MeanMetricWrapper):
    def __init__(
        self,
        reg_dim: int = 0,
        bool_dim: int = 0,
        ord_dim_tup: Tuple[int] = (0,),
        cat_dim_tup: Tuple[int] = (0,),
        group_weights: Optional[int] = None,
        name="VaeCategoricalGroupAverageAccuracy",
        dtype=None,
    ):

        if (reg_dim + bool_dim + sum(ord_dim_tup) + sum(cat_dim_tup)) <= 0:
            raise ValueError("Zero Dimensions supplied as shape of output")
        if sum(cat_dim_tup) <= 0:
            raise ValueError(
                "Zero Dimensions supplied as categorical input shape"
            )
        if group_weights is not None:
            if len(group_weights) != len(cat_dim_tup):
                raise ValueError(
                    f"Number of group weights {len(group_weights)} does not "
                    f"match number of categorical dimensions "
                    f"{len(cat_dim_tup)} "
                )

        def fn(y_true, y_pred):
            return vae_categorical_dims_accuracy(
                y_true,
                y_pred,
                reg_dim=reg_dim,
                bool_dim=bool_dim,
                ord_dim_tup=ord_dim_tup,
                cat_dim_tup=cat_dim_tup,
                group_weights=group_weights,
            )

        super(VaeCategoricalAvgAccuracy, self).__init__(
            fn,
            name=name,
            dtype=dtype,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):

        """Accumulates metric statistics.
        `y_true` and `y_pred` should have the same shape.

        Args:

        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
        sample_weight: Optional `sample_weight` acts as a
            coefficient for the metric. If a scalar is provided, then the metric is
            simply scaled by the given value. If `sample_weight` is a tensor of size
            `[batch_size]`, then the metric for each sample of the batch is rescaled
            by the corresponding element in the `sample_weight` vector. If the shape
            of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted
            to this shape), then each metric element of `y_pred` is scaled by the
            corresponding value of `sample_weight`. (Note on `dN-1`: all metric
            functions reduce by 1 dimension, usually the last axis (-1)).

        Returns:
        Update op.
        """
        y_true = math_ops.cast(y_true, self._dtype)
        # if type(y_pred) == dict:
        y_pred = {k: math_ops.cast(v, self._dtype) for k, v in y_pred.items()}
        # else:
        #    y_pred = math_ops.cast(y_pred, self._dtype)
        [
            y_true,
            y_pred,
        ], sample_weight = metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [y_true, y_pred], sample_weight
        )
        # y_pred, y_true = tf_losses_utils.squeeze_or_expand_dimensions(
        #    y_pred, y_true)

        ag_fn = autograph.tf_convert(self._fn, ag_ctx.control_status_ctx())
        matches = ag_fn(y_true, y_pred, **self._fn_kwargs)
        return super(MeanMetricWrapper, self).update_state(
            matches, sample_weight=sample_weight
        )


def vae_elbo(
    y_true,
    y_pred,
):
    ...


class VaeLoss(MeanMetricWrapper):
    ...


# def vae_categorical_dims_precision()

# class VaeCategoricalAccuracy

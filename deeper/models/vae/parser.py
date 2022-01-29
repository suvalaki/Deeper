from __future__ import annotations
import tensorflow as tf

from deeper.models.vae.network import VaeNet
from deeper.models.gan.base_getter import (
    BaseGanFakeOutputGetter,
    BaseGanRealOutputGetter,
)


class InputParser(BaseGanRealOutputGetter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        y_pred: VaeNet.Output,
        training: bool = False,
    ) -> tf.Tensor:
        return y


class OutputParser(BaseGanFakeOutputGetter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(
        self,
        y_pred: VaeNet.Output,
        training: bool = False,
    ) -> tf.Tensor:
        return tf.concat(
            [
                y_pred.px_g_z.regression,
                y_pred.px_g_z.binary,
                y_pred.px_g_z.ord_groups_concat,
                y_pred.px_g_z.categorical_groups_concat,
            ],
            axis=-1,
        )

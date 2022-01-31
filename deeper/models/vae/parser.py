from __future__ import annotations
import tensorflow as tf

from deeper.models.vae.network import VaeNet
from deeper.models.gan.base_getter import (
    BaseGanFakeOutputGetter,
    BaseGanRealOutputGetter,
)
from deeper.models.adversarial_autoencoder.base_getter import (
    AdversarialAutoencoderReconstructionLossGetter,
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


class LatentPriorParser(BaseGanRealOutputGetter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        y_pred: VaeNet.Output,
        training: bool = False,
    ) -> tf.Tensor:
        # Just generate a new value from scratch?
        return tf.random.normal(shape=tf.shape(y_pred.qz_g_x.sample))


class LatentPosteriorParser(BaseGanFakeOutputGetter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(
        self,
        y_pred: VaeNet.Output,
        training: bool = False,
    ) -> tf.Tensor:
        return y_pred.qz_g_x.sample


class ReconstructionOnlyLossOutputParser(AdversarialAutoencoderReconstructionLossGetter):
    def call(
        self,
        lossnet_out: VaeLossNet.Output,
        training=False,
    ) -> tf.Tensor:
        return -lossnet_out.scaled_l_pxgz

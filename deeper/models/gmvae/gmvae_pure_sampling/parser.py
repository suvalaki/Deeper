from __future__ import annotations
import tensorflow as tf

from deeper.models.gmvae.gmvae_pure_sampling.network import GumbleGmvaeNet
from deeper.models.gmvae.gmvae_pure_sampling.network_loss import (
    GumbleGmvaeNetLossNet,
)
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
        y_pred: GumbleGmvaeNet.Output,
        training: bool = False,
    ) -> tf.Tensor:
        return tf.concat(
            [
                y_pred.marginal.px_g_zy.regression,
                y_pred.marginal.px_g_zy.binary,
                y_pred.marginal.px_g_zy.ord_groups_concat,
                y_pred.marginal.px_g_zy.categorical_groups_concat,
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
        y_pred: GumbleGmvaeNet.Output,
        training: bool = False,
    ) -> tf.Tensor:
        return y_pred.marginal.pz_g_y.sample


class LatentPosteriorParser(BaseGanFakeOutputGetter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(
        self,
        y_pred: GumbleGmvaeNet.Output,
        training: bool = False,
    ) -> tf.Tensor:
        return y_pred.marginal.qz_g_xy.sample


class ReconstructionOnlyLossOutputParser(AdversarialAutoencoderReconstructionLossGetter):
    def call(
        self,
        lossnet_out: GumbleGmvaeNetLossNet.Output,
        training=False,
    ) -> tf.Tensor:
        return -lossnet_out.scaled_l_pxgzy

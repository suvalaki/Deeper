from __future__ import annotations
import tensorflow as tf

from deeper.models.gmvae.gmvae_marginalised_categorical.network import (
    StackedGmvaeNet,
)
from deeper.models.gmvae.gmvae_marginalised_categorical.network_loss import (
    StackedGmvaeLossNet,
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
        y_pred: StackedGmvaeNet.Output,
        training: bool = False,
    ) -> tf.Tensor:
        return tf.reduce_sum(
            tf.stack(
                [
                    y_pred.qy_g_x.probs[:, i, None]
                    * tf.concat(
                        [
                            marginal.px_g_zy.regression,
                            marginal.px_g_zy.binary,
                            marginal.px_g_zy.ord_groups_concat,
                            marginal.px_g_zy.categorical_groups_concat,
                        ],
                        axis=-1,
                    )
                    for i, marginal in enumerate(y_pred.marginals)
                ],
                axis=0,
            ),
            axis=0,
        )


class LatentPriorParser(BaseGanRealOutputGetter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        y_pred: StackedGmvaeNet.Output,
        training: bool = False,
    ) -> tf.Tensor:
        return tf.reduce_sum(
            tf.stack(
                [
                    y_pred.py[:, i, None] * marginal.pz_g_y.sample
                    for i, marginal in enumerate(y_pred.marginals)
                ],
                axis=0,
            ),
            axis=0,
        )


class LatentPosteriorParser(BaseGanFakeOutputGetter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(
        self,
        y_pred: StackedGmvaeNet.Output,
        training: bool = False,
    ) -> tf.Tensor:
        return tf.reduce_sum(
            tf.stack(
                [
                    y_pred.qy_g_x.probs[:, i, None] * marginal.qz_g_xy.sample
                    for i, marginal in enumerate(y_pred.marginals)
                ],
                axis=0,
            ),
            axis=0,
        )


class ReconstructionOnlyLossOutputParser(AdversarialAutoencoderReconstructionLossGetter):
    def call(
        self,
        lossnet_out: StackedGmvaeLossNet.Output,
        training=False,
    ) -> tf.Tensor:
        return -lossnet_out.scaled_l_pxgzy

from __future__ import annotations
import tensorflow as tf

from deeper.models.autoencoder.network import AutoencoderNet
from deeper.models.gan.base_getter import (
    BaseGanFakeOutputGetter,
    BaseGanRealOutputGetter,
)
from deeper.models.adversarial_autoencoder.base_getter import (
    AdversarialAutoencoderReconstructionLossGetter,
)


class OutputParser(BaseGanFakeOutputGetter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(
        self,
        y_pred: AutoencoderNet.Output,
        training: bool = False,
    ) -> tf.Tensor:
        return tf.concat(
            [
                y_pred.reconstruction.regression,
                y_pred.reconstruction.binary,
                y_pred.reconstruction.ord_groups_concat,
                y_pred.reconstruction.categorical_groups_concat,
            ],
            axis=-1,
        )

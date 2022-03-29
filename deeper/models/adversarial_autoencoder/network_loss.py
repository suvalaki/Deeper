from __future__ import annotations

import tensorflow as tf
from typing import NamedTuple, Union

from deeper.models.gan.network import (
    GanNet,
    GanGenerativeNet,
    GanDescriminativeNet,
)

from deeper.models.adversarial_autoencoder.network import AdversarialAutoencoderNet

from deeper.models.gan.network_loss import GanLossNet


class AdverasrialAutoencoderLossNet(GanLossNet):
    class Input(NamedTuple):
        adversarial: AdversarialAutoencoderNet.Output
        autoencoder: NamedTuple

        @classmethod
        def from_output(
            cls,
            network,
            autoencoder_lossnet,
            y_true,
            y_pred: AdversarialAutoencoderNet.Output,
            weight,
        ):
            y_true = tf.cast(y_true, dtype=network.dtype)
            y_split = network.generatornet.split_outputs(y_true)
            return cls(
                y_pred,
                autoencoder_lossnet.Input.from_output(y_split, y_pred.reconstruction, weight),
            )

    class Output(NamedTuple):
        generative: tf.Tesnor
        descriminative: tf.Tensor
        reconstruciton: tf.Tensor

    # Reconstruction + get gan to fool descrim
    def __init__(self, config, ae_net, **kwargs):
        super().__init__(**kwargs)
        self.network = ae_net
        self.generator_lossnet = config.generator.get_lossnet_type()()
        self.reconstruction_parser = config.generator.get_adversarialae_recon_loss_getter()()

    def call(
        self,
        inputs,
        training: bool = False,
    ):

        reconstruction = self.reconstruction_parser(
            self.generator_lossnet(inputs.autoencoder, training),
            training,
        )

        return AdverasrialAutoencoderLossNet.Output(
            self.gen_lossnet(inputs, inputs.adversarial.generative, training),
            self.descrim_lossnet(inputs, inputs.adversarial.descriminative, training),
            reconstruction,
        )

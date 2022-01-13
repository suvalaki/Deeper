from __future__ import annotations
import tensorflow as tf

from typing import Union, NamedTuple

from deeper.models.vae import VaeNet, VaeLossNet, VaeLatentParser
from deeper.models.gmvae.gmvae_marginalised_categorical import (
    StackedGmvaeNet,
    StackedGmvaeLossNet,
    StackedGmvaeLatentParser,
)
from deeper.models.gmvae.gmvae_pure_sampling import (
    GumbleGmvaeNet,
    GumbleGmvaeNetLossNet,
    GumbleGmvaeLatentParser,
)

ConfigType = Union[VaeNet.Config, StackedGmvaeNet.Config, GumbleGmvaeNet.Config]


class GeneralisedAutoencoderNet(tf.keras.layers.Layer):
    class Output(NamedTuple):
        network: NamedTuple
        losses: NamedTuple
        latent: tf.Tensor

    @staticmethod
    def _network_switch(config: ConfigType):

        # check in decreasing inheritance order
        if isinstance(config, StackedGmvaeNet.Config):
            return (
                StackedGmvaeNet(config),
                StackedGmvaeLossNet(),
                StackedGmvaeLatentParser(),
            )
        elif isinstance(config, GumbleGmvaeNet.Config):
            return (
                GumbleGmvaeNet(config),
                GumbleGmvaeNetLossNet(),
                GumbleGmvaeLatentParser(),
            )
        elif isinstance(config, VaeNet.Config):
            return VaeNet(config), VaeLossNet(prefix="loss"), VaeLatentParser()

    def __init__(self, config: ConfigType, **kwargs):
        super().__init__(**kwargs)
        self.network, self.lossnet, self.latent_parser = self._network_switch(config)

    @tf.function
    def call(self, x, y, weight=None, training=False):

        if weight is None:
            weight = self.lossnet.InputWeight()

        y_pred = self.network(x, training)
        losses = self.loss_fn(y, y_pred, weight, training)
        latent = self.latent_parser(y_pred)
        return self.Output(y_pred, losses, latent)

    @tf.function
    def loss_fn(
        self,
        y_true,
        y_pred,
        weight,
        training=False,
    ) -> VaeLossNet.output:

        y_true = tf.cast(y_true, dtype=self.dtype)
        y_split = self.network.split_outputs(y_true)
        loss = self.lossnet.Output(
            *[
                tf.reduce_mean(x)
                for x in self.lossnet(
                    self.lossnet.Input.from_output(
                        y_split,
                        y_pred,
                        weight,
                    ),
                    training,
                )
            ]
        )

        return loss

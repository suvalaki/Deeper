from __future__ import annotations
import tensorflow as tf

from typing import Union, NamedTuple

from deeper.utils.tf.experimental.extension_type import ExtensionTypeIterableMixin
from deeper.models.vae import Vae, VaeNet, VaeLossNet, VaeLatentParser
from deeper.models.gmvae.gmvae_marginalised_categorical import (
    StackedGmvae,
    StackedGmvaeNet,
    StackedGmvaeLossNet,
    StackedGmvaeLatentParser,
)
from deeper.models.gmvae.gmvae_pure_sampling import (
    GumbleGmvae,
    GumbleGmvaeNet,
    GumbleGmvaeNetLossNet,
    GumbleGmvaeLatentParser,
)

ConfigType = Union[VaeNet.Config, StackedGmvaeNet.Config, GumbleGmvaeNet.Config]
ModelConfigType = Union[Vae.Config, StackedGmvae.Config, GumbleGmvae.Config]


class GeneralisedAutoencoderNet(tf.keras.layers.Layer):
    class Output(tf.experimental.ExtensionType, ExtensionTypeIterableMixin):
        network: tf.experimental.ExtensionType
        losses: tf.experimental.ExtensionType
        latent: tf.Tensor

    @staticmethod
    def network_type_switch(config: ConfigType, **kwargs):

        return (
            config.get_network_type(),
            config.get_lossnet_type(),
            config.get_latent_parser_type(),
            config.get_model_type(),
        )

    @staticmethod
    def _network_switch(config: ConfigType, **kwargs):

        (
            network_t,
            lossnet_t,
            latent_t,
            model_t,
        ) = GeneralisedAutoencoderNet.network_type_switch(config)

        return (
            network_t(config, **kwargs),
            lossnet_t(**kwargs),
            latent_t(**kwargs),
        )

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

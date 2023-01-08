from __future__ import annotations
import tensorflow as tf

from typing import Tuple, Sequence, Optional, NamedTuple
from pydantic.dataclasses import dataclass

from deeper.layers.data_splitter import SplitCovariates
from deeper.layers.categorical import CategoricalEncoder
from deeper.probability_layers.gumble_softmax import GumbleSoftmaxLayer
from deeper.models.gmvae.marginalvae import MarginalGmVaeNet
from deeper.models.gmvae.base import GmvaeNetBase
from deeper.models.gmvae.gmvae_pure_sampling.utils import GumbleGmvaeTypeGetter
from deeper.utils.tf.experimental.extension_type import ExtensionTypeIterableMixin

from pydantic import BaseModel


# Extra class to collect encoder vars
class GumbleGmvaeNetEncoder(tf.keras.layers.Layer):
    def __init__(self, graph_qy_g_x_ohe, ae_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_qy_g_x_ohe = graph_qy_g_x_ohe
        self.ae_encoder = ae_encoder

    @tf.function
    def call(self, inputs, training=False):

        x, temperature = inputs
        x = tf.cast(x, self.dtype)
        temperature = tf.cast(temperature, self.dtype)
        qy_g_x = self.graph_qy_g_x(x, training=training)
        qy_g_x_ohe = self.graph_qy_g_x_ohe(qy_g_x.logits, temperature)
        return self.ae_encoder([x, qy_g_x_ohe], training)


class GumbleGmvaeNet(GmvaeNetBase):
    class Config(GumbleGmvaeTypeGetter, GmvaeNetBase.Config):
        ...

    class Output(tf.experimental.ExtensionType, ExtensionTypeIterableMixin, GumbleGmvaeTypeGetter):
        py: tf.Tensor
        qy_g_x: CategoricalEncoder.Output
        qy_gumble_one_hot_sample: tf.Tensor
        marginal: MarginalGmVaeNet.Output

        @property
        def encoder(self):
            return self.marginal.qz_g_xy

        @property
        def decoder(self):
            return self.marginal.px_g_zy

    def __init__(self, config: GumbleGmvaeNet.Config, **kwargs):
        super(GumbleGmvaeNet, self).__init__(**kwargs)
        self.components = config.components
        self.graph_qy_g_x = CategoricalEncoder(
            latent_dimension=config.components,
            embedding_dimensions=config.cat_embedding_dimensions,
            embedding_activation=config.embedding_activations,
            bn_before=config.bn_before,
            bn_after=config.bn_after,
            epsilon=config.categorical_epsilon,
            embedding_kernel_initializer=config.cat_embedding_kernel_initializer,
            embedding_bias_initializer=config.cat_embedding_bias_initializer,
            latent_kernel_initialiazer=config.cat_latent_kernel_initialiazer,
            latent_bias_initializer=config.cat_latent_bias_initializer,
            name="ygx",
        )
        self.graph_qy_g_x_ohe = GumbleSoftmaxLayer()
        self.graph_marginal_autoencoder = MarginalGmVaeNet(config, **kwargs)

        # shorthand
        self._encoder = GumbleGmvaeNetEncoder(
            self.graph_qy_g_x_ohe, self.graph_marginal_autoencoder.encoder
        )

    @property
    def encoder(self):
        # return self.graph_marginal_autoencoder
        return self._encoder

    @property
    def decoder(self):
        return self.graph_marginal_autoencoder.decoder

    @tf.function
    def call(self, inputs, training=False):

        x, temperature = inputs
        x = tf.cast(x, self.dtype)
        temperature = tf.cast(temperature, self.dtype)
        py = tf.cast(
            tf.fill(
                (tf.shape(x)[0], self.components),
                1.0 / self.components,
                name="prob",
            ),
            self.dtype,
        )

        qy_g_x = self.graph_qy_g_x(x, training=training)
        qy_g_x_ohe = self.graph_qy_g_x_ohe(qy_g_x.logits, temperature)
        marginal = self.graph_marginal_autoencoder([x, qy_g_x_ohe], training)

        return self.Output(py, qy_g_x, qy_g_x_ohe, marginal)

    @tf.function
    def split_outputs(self, y) -> SplitCovariates:
        return self.graph_marginal_autoencoder.split_outputs(y)

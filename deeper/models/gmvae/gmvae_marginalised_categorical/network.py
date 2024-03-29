from __future__ import annotations
import tensorflow as tf
import numpy as np

from typing import Union, Tuple, Sequence, Optional, NamedTuple
from pydantic.dataclasses import dataclass

from deeper.layers.categorical import CategoricalEncoder
from deeper.probability_layers.gumble_softmax import GumbleSoftmaxLayer
from deeper.models.gmvae.marginalvae import MarginalGmVaeNet
from deeper.models.gmvae.base import GmvaeNetBase
from deeper.models.gmvae.gmvae_marginalised_categorical.utils import (
    StackedGmvaeTypeGetter,
)
from deeper.models.vae import (
    VaeEncoderNet,
    VaeReconstructionNet,
)
from deeper.ops.reduce import weighted_average

from pydantic import BaseModel
from deeper.utils.tf.experimental.extension_type import ExtensionTypeIterableMixin


class StackedGmvaeNet(GmvaeNetBase):
    class Config(StackedGmvaeTypeGetter, GmvaeNetBase.Config):
        ...

    class Output(tf.experimental.ExtensionType, ExtensionTypeIterableMixin, StackedGmvaeTypeGetter):
        py: tf.Tensor
        qy_g_x: CategoricalEncoder.Output
        marginals: Tuple[MarginalGmVaeNet.Output, ...]

        # an addotional average model to mock a decoder in helper methods. 
        # maybe this should be a dynamic prop instead?
        encoder: VaeEncoderNet.Output
        decoder: VaeReconstructionNet.Output


    def __init__(self, config: GmvaeNet.Config, **kwargs):
        super(StackedGmvaeNet, self).__init__(**kwargs)
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
        self.graph_marginal_autoencoder = MarginalGmVaeNet(config, **kwargs)

    @property
    def encoder(self):
        return self.graph_marginal_autoencoder.encoder
        # raise NotImplementedError()

    @property
    def decoder(self):
        return self.graph_marginal_autoencoder.decoder
        # raise NotImplementedError()

    @tf.function
    def call(self, inputs, training=False):

        x = inputs
        x = tf.cast(x, self.dtype)
        py = tf.cast(
            tf.fill(
                (tf.shape(x)[0], self.components),
                1.0 / self.components,
                name="prob",
            ),
            self.dtype,
        )

        qy_g_x = self.graph_qy_g_x(x, training=training)
        y_ = tf.cast(tf.fill(tf.stack([tf.shape(x)[0], self.components]), 0.0), x.dtype)
        marginals = [None] * self.components
        for i in range(self.components):
            with tf.name_scope("mixture_{}".format(i)):
                y_ohe = tf.add(
                    y_,
                    tf.constant(
                        np.eye(self.components)[i],
                        dtype=x.dtype,
                        name="y_one_hot_{}".format(i),
                    ),
                    name="hot_at_{}".format(i),
                )
                marginals[i] = self.graph_marginal_autoencoder([x, y_ohe], training)

        probs = qy_g_x.probs
        averaged = MarginalGmVaeNet.Output.reduce(marginals, lambda x: weighted_average(x, probs))

        return StackedGmvaeNet.Output(py, qy_g_x, marginals, averaged.encoder, averaged.decoder)

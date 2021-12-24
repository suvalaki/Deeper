from __future__ import annotations
import tensorflow as tf
import numpy as np
from deeper.utils.scope import Scope
from typing import Optional, Sequence

tfk = tf.keras
Layer = tfk.layers.Layer
Model = tfk.Model

from dataclasses import dataclass


@dataclass
class BaseEncoderConfig:
    latent_dim: int = None
    embedding_dimensions: Sequence[int] = None
    var_scope: str = "encoder"
    bn_before: bool = False
    bn_after: bool = False


class Encoder(Layer):
    @dataclass
    class Config(BaseEncoderConfig):
        activation: tf.keras.layers.Activation = tf.nn.relu
        embedding_kernel_initializer = tf.initializers.glorot_uniform()
        embedding_bias_initializer = tf.initializers.zeros()
        latent_kernel_initialiazer = tf.initializers.glorot_uniform()
        latent_bias_initializer = tf.initializers.zeros()
        embedding_dropout: Optional[float] = None
        embedding_kernel_regularizer = None
        embedding_bias_regularizer = None
        latent_kernel_regularizer = None
        latent_bias_regularizer = None

    @classmethod
    def from_config(cls, config: Encoder.Config, **kwargs):
        return cls(**config, **kwargs)

    def __init__(
        self,
        latent_dim: int,
        embedding_dimensions: Sequence,
        activation,
        var_scope: str = "encoder",
        bn_before: bool = False,
        bn_after: bool = False,
        embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        embedding_bias_initializer=tf.initializers.zeros(),
        latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        latent_bias_initializer=tf.initializers.zeros(),
        embedding_dropout: Optional[float] = None,
        embedding_kernel_regularizer=tf.keras.regularizers.l2(),
        embedding_bias_regularizer=tf.keras.regularizers.l2(),
        latent_kernel_regularizer=None,
        latent_bias_regularizer=None,
        **kwargs
    ):

        # Activate V1 Type behaviour. Layer takes the dtype of its inputs
        V1_PARMS = {"autocast": False}

        Layer.__init__(self, **V1_PARMS, **kwargs)
        # Scope.__init__(self, var_scope)
        self.latent_dim = latent_dim
        self.em_dim = embedding_dimensions

        # embeddings
        self.n_em = len(embedding_dimensions)
        self.embeddings = [None] * self.n_em
        self.embeddings_bn_before = [None] * self.n_em
        self.embeddings_bn_after = [None] * self.n_em
        self.activation = activation
        self.bn_before = bn_before
        self.bn_after = bn_after
        self.dropout_rate = (
            embedding_dropout if embedding_dropout is not None else 0.0
        )
        self.dropout = [None] * self.n_em

        for i, em in enumerate(self.em_dim):
            with tf.name_scope("embedding_{}".format(i)):

                self.embeddings[i] = tfk.layers.Dense(
                    units=em,
                    activation=None,
                    use_bias=True,
                    kernel_initializer=embedding_kernel_initializer,
                    bias_initializer=embedding_bias_initializer,
                    kernel_regularizer=embedding_kernel_regularizer,
                    bias_regularizer=embedding_bias_regularizer,
                    name="dense",
                    **V1_PARMS,
                )
                if self.bn_before:
                    self.embeddings_bn_before[
                        i
                    ] = tfk.layers.BatchNormalization(
                        axis=-1,
                        name="bn_before",
                        renorm=True,
                        **V1_PARMS,
                    )

                if self.bn_after:
                    self.embeddings_bn_after[
                        i
                    ] = tfk.layers.BatchNormalization(
                        axis=-1,
                        name="bn_after",
                        renorm=True,
                        **V1_PARMS,
                    )

                if self.dropout_rate > 0.0:
                    self.dropout[i] = tfk.layers.Dropout(
                        self.dropout_rate,
                        name="dropout",
                        **V1_PARMS,
                    )

        self.latent_bn = tfk.layers.BatchNormalization(
            axis=-1,
            name="latent_bn",
            **V1_PARMS,
        )
        self.latent = tfk.layers.Dense(
            units=self.latent_dim,
            activation=None,
            use_bias=True,
            kernel_initializer=latent_kernel_initialiazer,
            bias_initializer=latent_bias_initializer,
            kernel_regularizer=latent_kernel_regularizer,
            bias_regularizer=latent_bias_regularizer,
            name="latent_dense",
            **V1_PARMS,
        )

    @tf.function
    def call(self, inputs, training=False):
        """Define the computational flow

        Parameters
        ----------
        inpits: Tensor, input
        training: bool, whether to apply training calculation through batch
            normalisation.
        ldr: allowable lower depth range. call through embedding layers will
            include layers with a lesser than or equal to index value to this.
            The embedding layers between aldr and audr will be skipped.
        audr: allowable upper depth range. call through embedding layers will
            include layers with a higher than or equal to index value to this.

        """
        x = inputs

        for i, (embedding, bn_before, bn_after, dropout) in enumerate(
            zip(
                self.embeddings,
                self.embeddings_bn_before,
                self.embeddings_bn_after,
                self.dropout,
            )
        ):
            with tf.name_scope("embedding_{}".format(i)):
                x = embedding(x)
                if self.bn_before:
                    x = bn_before(x, training=training)
                x = self.activation(x)
                if self.bn_after:
                    x = bn_after(x, training=training)
                if self.dropout_rate > 0.0:
                    x = dropout(x, training=training)

        x = self.latent(x)
        return x

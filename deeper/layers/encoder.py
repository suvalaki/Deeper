from __future__ import annotations
import tensorflow as tf
import numpy as np
from deeper.utils.scope import Scope
from typing import Optional, Sequence, Union

tfk = tf.keras
Layer = tfk.layers.Layer
Model = tfk.Model

from dataclasses import dataclass
from pydantic import BaseModel

from deeper.optimizers.automl.tunable_types import (
    TunableModelMixin,
    TunableBoolean,
    TunableActivation,
    OptionalTunableL1L2Regulariser,
    OptionalTunableDropout,
)


class BaseEncoderConfig(TunableModelMixin):
    latent_dim: int = None
    embedding_dimensions: Sequence[int] = []
    var_scope: str = "encoder"
    bn_before: TunableBoolean = TunableBoolean(False)
    bn_after: bool = False

    class Config:
        arbitrary_types_allowed = True
        smart_union = True


BaseEncoderConfig.update_forward_refs()


class Encoder(Layer):
    class Config(BaseEncoderConfig):
        activation: tf.keras.layers.Activation = TunableActivation("relu")
        embedding_kernel_initializer: Union[
            str, tf.keras.initializers.Initializer
        ] = tf.initializers.glorot_uniform()
        embedding_bias_initializer: Union[
            str, tf.keras.initializers.Initializer
        ] = tf.initializers.zeros()
        latent_kernel_initialiazer: Union[
            str, tf.keras.initializers.Initializer
        ] = tf.initializers.glorot_uniform()
        latent_bias_initializer: Union[
            str, tf.keras.initializers.Initializer
        ] = tf.initializers.zeros()
        input_dropout: Optional[float] = OptionalTunableDropout()
        embedding_dropout: Optional[float] = OptionalTunableDropout()
        embedding_kernel_regularizer: Optional[
            tf.keras.regularizers.Regularizer
        ] = OptionalTunableL1L2Regulariser(0.0, 0.0)
        embedding_bias_regularizer: Optional[
            tf.keras.regularizers.Regularizer
        ] = OptionalTunableL1L2Regulariser(0.0, 0.0)
        embedding_activity_regularizer: Optional[tf.keras.regularizers.Regularizer] = None
        latent_kernel_regularizer: Optional[
            tf.keras.regularizers.Regularizer
        ] = OptionalTunableL1L2Regulariser(0.0, 0.0)
        latent_bias_regularizer: Optional[
            tf.keras.regularizers.Regularizer
        ] = OptionalTunableL1L2Regulariser(0.0, 0.0)
        latent_activity_regularizer: Optional[tf.keras.regularizers.Regularizer] = None

    Config.update_forward_refs()

    @classmethod
    def from_config(cls, config: Encoder.Config, **kwargs):
        return cls(**dict(config), **kwargs)

    def __init__(
        self,
        latent_dim: int,
        embedding_dimensions: Sequence[int],
        activation,
        var_scope: str = "encoder",
        bn_before: bool = False,
        bn_after: bool = False,
        embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        embedding_bias_initializer=tf.initializers.zeros(),
        latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        latent_bias_initializer=tf.initializers.zeros(),
        input_dropout: Optional[float] = None,
        embedding_dropout: Optional[float] = None,
        embedding_kernel_regularizer=None,  # tf.keras.regularizers.l2(),
        embedding_bias_regularizer=None,  # tf.keras.regularizers.l2(),
        embedding_activity_regularizer=None,  # tf.keras.regularizers.l2(),
        latent_kernel_regularizer=None,
        latent_bias_regularizer=None,
        latent_activity_regularizer=None,  # tf.keras.regularizers.l2(),
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
        self.input_dropout_rate = input_dropout
        self.dropout_rate = embedding_dropout if embedding_dropout is not None else 0.0
        self.dropout = [None] * self.n_em

        if self.input_dropout_rate is not None:
            self.input_dropout = tf.keras.layers.Dropout(self.input_dropout_rate)

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
                    activity_regularizer=embedding_activity_regularizer,
                    name="dense",
                    **V1_PARMS,
                )
                if self.bn_before:
                    self.embeddings_bn_before[i] = tfk.layers.BatchNormalization(
                        axis=-1,
                        name="bn_before",
                        renorm=True,
                        **V1_PARMS,
                    )

                if self.bn_after:
                    self.embeddings_bn_after[i] = tfk.layers.BatchNormalization(
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
            activity_regularizer=latent_activity_regularizer,
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

        if self.input_dropout_rate is not None:
            x = self.input_dropout(x, training=training)

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
                x = self.activation(x, training=training)
                if self.bn_after:
                    x = bn_after(x, training=training)
                if self.dropout_rate > 0.0:
                    x = dropout(x, training=training)

        x = self.latent(x)
        return x

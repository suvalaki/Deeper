import tensorflow as tf
import numpy as np
from deeper.utils.scope import Scope

tfk = tf.keras
Layer = tfk.layers.Layer
Model = tfk.Model


class Encoder(Layer, Scope):
    def __init__(
        self,
        latent_dim,
        embedding_dimensions,
        activation,
        var_scope="encoder",
        bn_before=False,
        bn_after=False,
        embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        embedding_bias_initializer=tf.initializers.zeros(),
        latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        latent_bias_initializer=tf.initializers.zeros(),
        embedding_dropout=0.0,
    ):

        # Activate V1 Type behaviour. Layer takes the dtype of its inputs
        V1_PARMS = {"autocast": False}

        Layer.__init__(self, **V1_PARMS)
        Scope.__init__(self, var_scope)
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
        self.dropout_rate = embedding_dropout
        self.dropout = [None] * self.n_em

        for i, em in enumerate(self.em_dim):
            with tf.name_scope('embedding_{}'.format(i)):
                self.embeddings[i] = (
                    tfk.layers.Dense(
                        units=em,
                        activation=None,
                        use_bias=True,
                        kernel_initializer=embedding_kernel_initializer,
                        bias_initializer=embedding_bias_initializer,
                        name='dense',
                        **V1_PARMS,
                    )
                )
                if self.bn_before:
                    self.embeddings_bn_before[i] = (
                        tfk.layers.BatchNormalization(
                            axis=-1,
                            name='bn_before',
                            renorm=True,
                            **V1_PARMS,
                        )
                    )

                if self.bn_after:
                    self.embeddings_bn_after[i] = (
                        tfk.layers.BatchNormalization(
                            axis=-1,
                            name="bn_after",
                            renorm=True,
                            **V1_PARMS,
                        )
                    )

                if self.dropout_rate > 0.0:
                    self.dropout[i] = (tfk.layers.Dropout(
                        self.dropout_rate, 
                        name='dropout',
                        **V1_PARMS,
                    ))

        self.latent_bn = tfk.layers.BatchNormalization(
            axis=-1, name=self.v_name("latent_bn"), 
            **V1_PARMS,
        )
        self.latent = tfk.layers.Dense(
            units=self.latent_dim,
            activation=None,
            use_bias=True,
            kernel_initializer=latent_kernel_initialiazer,
            bias_initializer=latent_bias_initializer,
            name=self.v_name("latent_dense"),
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

        for i in range(self.n_em):
            with tf.name_scope('embedding_{}'.format(i)):
                x = self.embeddings[i](x)
                if self.bn_before:
                    x = self.embeddings_bn_before[i](x, training=training)
                x = self.activation(x)
                if self.bn_after:
                    x = self.embeddings_bn_after[i](x, training=training)
                if self.dropout_rate > 0.0:
                    x = self.dropout[i](x, training=training)
        x = self.latent(x)
        return x

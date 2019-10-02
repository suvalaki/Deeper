import tensorflow as tf 
import numpy as np 
from deeper.utils.scope import Scope

tfk = tf.keras
Layer = tfk.layers.Layer

class Encoder(Layer, Scope):
    def __init__(
        self, 
        latent_dim, 
        embedding_dimensions, 
        activation,
        var_scope='encoder', 
        bn_before=False,
        bn_after=False,
        embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        embedding_bias_initializer=tf.initializers.zeros(),
        latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        latent_bias_initializer=tf.initializers.zeros()
    ):
        Layer.__init__(self)
        Scope.__init__(self, var_scope)
        self.latent_dim = latent_dim
        self.em_dim = embedding_dimensions

        # embeddings
        self.embeddings = []
        self.embeddings_bn_before = []
        self.embeddings_bn_after = []
        self.activation = activation
        self.bn_before = bn_before
        self.bn_after = bn_after
        
        for i,em in enumerate(self.em_dim):
            self.embeddings.append(
                tfk.layers.Dense(
                    units=em,
                    activation=None,
                    use_bias=True,
                    kernel_initializer=embedding_kernel_initializer,
                    bias_initializer=embedding_bias_initializer,
                    name=self.v_name('embedding_{}_dense'.format(i))
                )
            )
            if self.bn_before:
                self.embeddings_bn_before.append(
                    tfk.layers.BatchNormalization(
                        axis=-1, 
                        name=self.v_name('embedding_{}_bn_before'.format(i))
                    )
                )
            else:
                self.embeddings_bn_before.append(None)

            if self.bn_after:
                self.embeddings_bn_after.append(
                    tfk.layers.BatchNormalization(
                        axis=-1,
                        name=self.v_name('embedding_{}_bn_after'.format(i))
                    )
                )
            else:
                self.embeddings_bn_after.append(None)

        self.latent_bn = tfk.layers.BatchNormalization(
            axis=-1, 
            name=self.v_name('latent_bn')
        )
        self.latent = tfk.layers.Dense(
            units=self.latent_dim,
            activation=None,
            use_bias=True,
            kernel_initializer=latent_kernel_initialiazer,
            bias_initializer=latent_bias_initializer,
            name=self.v_name('latent_dense')
        )

    @tf.function
    def call(self, inputs, training=False):
        """Define the computational flow"""
        x = inputs
        for em, bnb, bna in zip(
            self.embeddings, 
            self.embeddings_bn_before, 
            self.embeddings_bn_after
        ):
            x = em(x)
            if self.bn_before:
                x = bnb(x, training=training)
            x = self.activation(x)
            if self.bn_after:
                x = bna(x, training=training)
        x = self.latent(x)
        return x


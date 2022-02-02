from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import initializers
from typing import NamedTuple

from deeper.layers.encoder import Encoder

# from deeper.layers.conv2d_encoder import Conv2dEncoder, Conv2dDecoder
from deeper.utils.scope import Scope
from deeper.utils.function_helpers.decorators import inits_args

Layer = tf.keras.layers.Layer


class CategoricalEncoder(Layer):
    class Output(NamedTuple):
        logits: tf.Tensor
        probs: tf.Tensor
        argmax: tf.Tensor
        onehot: tf.Tensor

    def __init__(
        self,
        latent_dimension,
        embedding_dimensions,
        embedding_activation=tf.nn.relu,
        var_scope="cat_encoder",
        bn_before=False,
        bn_after=False,
        epsilon=0.0,
        embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        embedding_bias_initializer=tf.initializers.zeros(),
        latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        latent_bias_initializer=tf.initializers.zeros(),
        embedding_dropout=0.0,
        **kwargs
    ):
        Layer.__init__(self, **kwargs)

        self.latent_dimension = latent_dimension
        self.embedding_dimensions = embedding_dimensions
        self.embedding_activation = embedding_activation
        self.bn_before = bn_before
        self.bn_after = bn_after
        self.epsilon = epsilon

        self.logits_encoder = Encoder(
            latent_dim=self.latent_dimension,
            embedding_dimensions=self.embedding_dimensions,
            activation=self.embedding_activation,
            bn_before=self.bn_before,
            bn_after=self.bn_after,
            embedding_kernel_initializer=embedding_kernel_initializer,
            embedding_bias_initializer=embedding_bias_initializer,
            latent_kernel_initialiazer=latent_kernel_initialiazer,
            latent_bias_initializer=latent_bias_initializer,
            embedding_dropout=embedding_dropout,
        )

    @tf.function
    def call_logits(self, inputs, training=False):
        logits = self.logits_encoder(inputs, training)
        if self.epsilon > 0.0:
            maxval = np.log(1.0 - self.epsilon) - np.log(self.epsilon)
            logits = tf.compat.v2.clip_by_value(logits, -maxval, maxval)
        return logits

    @tf.function
    def _prob(self, logits):
        prob = tf.nn.softmax(logits, axis=-1, name="prob")
        if self.epsilon > 0.0:
            prob = tf.compat.v2.clip_by_value(
                prob, self.epsilon, 1 - self.epsilon, name="prob_clipped"
            )
        return prob

    @tf.function
    def call(self, inputs, y=None, training=False) -> CategoricalEncoder.Output:
        logits = self.logits_encoder(inputs, training)
        probs = self._prob(logits)
        argmax = tf.argmax(probs, axis=-1)
        oh = tf.one_hot(argmax, depth=self.latent_dimension)
        return CategoricalEncoder.Output(logits, probs, argmax, oh)

    @tf.function
    def prob(self, inputs, training=False):
        logits, probs = self.call(inputs, training)
        return probs

    @tf.function
    def entropy(self, x, y, training=False):
        logits = self.call_logits(x, training)
        ent = tf.nn.softmax_cross_entropy_with_logits(
            labels=y, logits=logits, name="entropy", axis=-1
        )
        return ent

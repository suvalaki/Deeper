import tensorflow as tf
import numpy as np
from deeper.layers.encoder import Encoder
from deeper.utils.scope import Scope

tfk = tf.keras

Layer = tfk.layers.Layer

class SigmoidEncoder(Layer, Scope):
    def __init__(
        self,
        latent_dimension, 
        embedding_dimensions, 
        embedding_activation=tf.nn.relu,
        var_scope='binary_encoder',
        bn_before=False,
        bn_after=False,
        epsilon=0.0,
    ):
        Layer.__init__(self)
        Scope.__init__(self, var_scope)

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
            var_scope=self.v_name('logits_encoder'), 
            bn_before=self.bn_before,
            bn_after=self.bn_after
        )

    @tf.function
    def call_logits(self, inputs, training=False):
        logits = self.logits_encoder(x, training)
        if self.epsilon > 0.0:
            maxval = np.log(1.0 - self.epsilon) - np.log(self.epsilon)
            logits = tf.compat.v2.clip_by_value(
                logits, -maxval, maxval, "clipped")
        return logits

    @tf.function
    def _prob(self, logits):
        prob = tf.nn.sigmoid(logits, name='probs')
        return prob

    @tf.function
    def prob(self, inputs, training=False):
        logits, probs = self.call(inputs, training)
        return probs

    @tf.function
    def entropy(self, x, y, training=False):
        logits = self.call_logits(x, training)
        ent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y, 
            logits=logits,
            name='entropy'
        )
        return ent

    @tf.function 
    def call(self, inputs, training=False, y=None):
        logits = self.logits_encoder(inputs, training)
        probs = self._prob(logits)
        if y is not None:
            ent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y, 
                logits=logits,
                name='entropy'
            )
        else:
            ent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=probs,
                logits=logits,
                name='entropy'
            )
        logprob = - tf.reduce_sum(ent, -1, name='logprob')
        prob = tf.math.exp(logprob, name='prob')
        return ent, logprob, prob 
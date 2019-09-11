import tensorflow as tf 
import numpy as np

from utils.scope import Scope
from layers.encoder import Encoder

tfk = tf.keras
Model = tfk.Model
Layer = tfk.layers.Layer

class NormalEncoder(Layer, Scope):
    def __init__(
        self, 
        latent_dimension, 
        embedding_dimensions, 
        var_scope='normal_encoder',
        bn_before=False,
        bn_after=False
    ):
        Model.__init__(self)
        Scope.__init__(self, var_scope)

        self.latent_dimension = latent_dimension
        self.embedding_dimensions = embedding_dimensions
        self.bn_before = bn_before
        self.bn_after = bn_after

        self.mu = Encoder(
            latent_dim=self.latent_dimension,
            embedding_dimensions=self.embedding_dimensions,
            activation=self.embedding_activation,
            var_scope=self.v_name('mu_encoder'), 
            bn_before=self.bn_before,
            bn_after=self.bn_after
        )
        self.logvar = Encoder(
            latent_dim=self.latent_dimension,
            embedding_dimensions=self.embedding_dimensions,
            activation=self.embedding_activation,
            var_scope=self.v_name('logvar_encoder'), 
            bn_before=self.bn_before,
            bn_after=self.bn_after
        )

    @tf.function
    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float64)
        mu = self.mu(x, training)
        logvar = self.logvar(x, training)

        # reparmeterisation trick
        r_norm = tf.random.normal( tf.shape(mu), mean=0., stddev=1.)
        sample = mu + r_norm * tf.math.sqrt(tf.exp(logvar))
        
        # Metrics for loss
        logprob = self.log_normal(sample, mu, tf.exp(logvar))
        prob = tf.exp(logprob)

        return sample, logprob, prob

    @staticmethod
    @tf.function
    def log_normal(x, mu, var, eps=0.0, axis=-1):
        if eps > 0.0:
            var = tf.add(var, eps, name='clipped_var')
        return -0.5 * tf.reduce_sum(
            tf.math.log(2 * np.pi) + tf.math.log(var) + tf.square(x - mu) / var, axis)


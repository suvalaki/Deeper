import tensorflow as tf 
import numpy as np

from deeper.utils.scope import Scope
from deeper.layers.encoder import Encoder

tfk = tf.keras
Layer = tfk.layers.Layer

class RandomNormalEncoder(Layer, Scope):
    def __init__(
        self, 
        latent_dimension, 
        embedding_dimensions, 
        embedding_activations=tf.nn.relu,
        var_scope='normal_encoder',
        bn_before=False,
        bn_after=False,
        epsilon=0.0
    ):
        Layer.__init__(self)
        Scope.__init__(self, var_scope)

        self.latent_dimension = latent_dimension
        self.embedding_dimensions = embedding_dimensions
        self.embedding_activation = embedding_activations
        self.bn_before = bn_before
        self.bn_after = bn_after
        self.epsilon = epsilon

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

    #@tf.function
    def call(self, inputs, training=False, outputs=None):
        mu = self.mu(inputs, training)
        logvar = self.logvar(inputs, training)

        if outputs is not None:
            sample = outputs
        else:
            # reparmeterisation trick
            r_norm = tf.cast(
                tf.random.normal( tf.shape(mu), mean=0., stddev=1.),
                inputs.dtype
            )
            sample = mu + r_norm * tf.math.sqrt(tf.exp(logvar))
        
        # Metrics for loss
        logprob = self.log_normal(sample, mu, tf.exp(logvar), self.epsilon)
        prob = tf.exp(logprob)

        return sample, logprob, prob

    @staticmethod
    @tf.function
    def log_normal(x, mu, var, eps=0.0, axis=-1):
        if eps > 0.0:
            var = tf.add(var, eps, name='clipped_var')
        return -0.5 * tf.reduce_sum(
            tf.math.log(2 * tf.cast(np.pi, x.dtype)) 
            + tf.math.log(var) + tf.square(x - mu) / var, axis)

    @tf.function
    def entropy(self, inputs, inputs_x, mu_x, var_x, training=False):
        """Compare mu and var against """
        mu = self.mu(inputs, training)
        logvar = self.logvar(inputs, training)
        entropy = (
            self.log_normal(inputs_x, mu, var, self.epslion)
            - self.log_normal(inputs_x, mu_x, var_x, self.epsilon)
        )
        return entropy




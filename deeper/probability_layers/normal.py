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
        epsilon=0.0,

        embedding_mu_kernel_initializer=tf.initializers.glorot_uniform(),
        embedding_mu_bias_initializer=tf.initializers.zeros(),
        latent_mu_kernel_initialiazer=tf.initializers.glorot_uniform(),
        latent_mu_bias_initializer=tf.initializers.zeros(),

        embedding_var_kernel_initializer=tf.initializers.glorot_uniform(),
        embedding_var_bias_initializer=tf.initializers.zeros(),
        latent_var_kernel_initialiazer=tf.initializers.glorot_uniform(),
        latent_var_bias_initializer=tf.initializers.ones()
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
            bn_after=self.bn_after,
            embedding_kernel_initializer=embedding_mu_kernel_initializer,
            embedding_bias_initializer=embedding_mu_bias_initializer,
            latent_kernel_initialiazer=latent_mu_kernel_initialiazer,
            latent_bias_initializer=latent_mu_bias_initializer
        )
        self.logvar = Encoder(
            latent_dim=self.latent_dimension,
            embedding_dimensions=self.embedding_dimensions,
            activation=self.embedding_activation,
            var_scope=self.v_name('logvar_encoder'), 
            bn_before=self.bn_before,
            bn_after=self.bn_after,
            embedding_kernel_initializer=embedding_var_kernel_initializer,
            embedding_bias_initializer=embedding_var_bias_initializer,
            latent_kernel_initialiazer=latent_var_kernel_initialiazer,
            latent_bias_initializer=latent_var_bias_initializer
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

        return sample, logprob, prob, mu, logvar

    @staticmethod
    @tf.function
    def log_normal(x, mu, var, eps=0.0, axis=-1):

        #x = tf.stop_gradient(x)

        if eps > 0.0:
            var = tf.add(var, eps, name='clipped_var')
        logprob = -0.5 * tf.reduce_sum(
            #tf.clip_by_value(
                tf.math.log(2 * tf.cast(np.pi, x.dtype)) 
                + tf.math.log(var) + tf.square(x - mu) / var
            #    ,np.log(1e-6)
            #    ,np.log(1 - 1e-6)
            #)    
            , axis
        )
        #logprob = tf.clip_by_value(logprob, np.log(1e-32), np.log(1-1e-32))
        return logprob


    @tf.function
    def entropy(self, inputs, inputs_x, mu_x, var_x, training=False, eps=0.0, axis=-1):
        """Compare mu and var against """
        mu = self.mu(inputs, training)
        logvar = self.logvar(inputs, training)
        var = tf.math.exp(logvar)  

        if eps > 0.0:
            var = tf.add(var, eps, name='clipped_var')
        
        if eps > 0.0:
            var_x = tf.add(var_x, eps, name='clipped_var_x')

        #entropy = -0.5*(
        #    tf.reduce_sum(
        #            tf.clip_by_value(
        #             tf.math.log(var) + tf.square(inputs_x - mu) / var
        #            - tf.math.log(var_x) - tf.square(inputs_x - mu_x) / var_x
        #            ,-1e3/tf.cast(tf.shape(inputs_x)[-1], inputs_x.dtype)
        #            , 1e3/tf.cast(tf.shape(inputs_x)[-1], inputs_x.dtype)
        #        )
        #        ,axis
        #    )
        #)

        entropy = (self.log_normal(inputs_x, mu, var, eps) 
            - self.log_normal(inputs_x, mu_x, var_x, eps))

        return entropy




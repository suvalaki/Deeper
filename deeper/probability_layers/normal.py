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
        epsilon=1e-8,

        embedding_mu_kernel_initializer=tf.initializers.glorot_uniform(),
        embedding_mu_bias_initializer=tf.initializers.zeros(),
        latent_mu_kernel_initialiazer=tf.initializers.glorot_normal(),
        latent_mu_bias_initializer=tf.initializers.zeros(),

        embedding_var_kernel_initializer=tf.initializers.glorot_uniform(),
        embedding_var_bias_initializer=tf.initializers.zeros(),
        latent_var_kernel_initialiazer=tf.initializers.glorot_uniform(),
        latent_var_bias_initializer=tf.initializers.ones()

        fixed_mu=None,
        fixed_var=None,
        connected_weights=True,

    ):
        Layer.__init__(self)
        Scope.__init__(self, var_scope)

        self.latent_dimension = latent_dimension
        self.embedding_dimensions = embedding_dimensions
        self.embedding_activation = embedding_activations
        self.bn_before = bn_before
        self.bn_after = bn_after
        self.epsilon = epsilon
        self.fixed_mu = fixed_mu
        self.fixed_var = fixed_var
        self.connected_weights = connected_weights

        if not self.connected_weights:
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
        else:
            self.mu_logvar = Encoder(
                latent_dim=2*self.latent_dimension,
                embedding_dimensions=[
                    2*x for x in self.embedding_dimensions
                ],
                activation=self.embedding_activation,
                var_scope=self.v_name('mu_encoder'), 
                bn_before=self.bn_before,
                bn_after=self.bn_after,
                embedding_kernel_initializer=embedding_mu_kernel_initializer,
                embedding_bias_initializer=embedding_mu_bias_initializer,
                latent_kernel_initialiazer=latent_mu_kernel_initialiazer,
                latent_bias_initializer=latent_mu_bias_initializer
            )
        

    @tf.function
    def call_parameters(self, inputs, training=False, apply_epsilon=True):
        """Return the parameters for this normal distribution

        Parameters:
        -----------
        inputs: Input vector to the 
        training: bool flag for whether the layer is training
        apply_epsilon: bool flag for whether toa djust the logvar by epsilon

        """

        if not self.connected_weights:
            mu = self.mu(inputs, training)
            logvar = self.logvar(inputs, training)
        else:
            mu_logvar = self.mu_logvar(inputs, training)
            mu, logvar = tf.split(mu_logvar, 2, axis=-1)

        if apply_epsilon:
            var = tf.exp(logvar) + self.epsilon
            logvar = tf.math.log(var)

        mu = tf.identity(mu, name=self.v_name('mu'))
        logvar = tf.identity(logvar, name=self.v_name('logvar'))

        return mu, logvar

    @tf.function
    def call_mu(self, inputs, training=False):
        mu, logvar = self.call_parameters(inputs, training)
        return mu

    @tf.function
    def call_logvar(self, inputs, training=False, apply_epsilon=True):
        mu, logvar = self.call_parameters(inputs, training, apply_epsilon)
        return logvar

    @tf.function 
    def _sample_fn(self, mu=0.0, var=1.0, name='sample'):

        # reparmeterisation trick
        r_norm = tf.cast(
            tf.random.normal( sample_shape, mean=0., stddev=1.),
            inputs.dtype
        )
        sample = tf.identity(
            mu + r_norm * tf.math.sqrt(var)
            ,name=self.v_name(name)
        )

        return sample

    @tf.function
    def sample(
        self, 
        inputs, 
        training=False, 
        apply_epsilon=False,
        name='sample'
    ):

        mu, logvar = self.call_parameters(inputs, training, apply_epsilon)
        sample = self._sample_fn(mu, logvar, name)
        return sample


    @tf.function
    def call(self, inputs, training=False, outputs=None, apply_epsilon=True):

        mu, logvar = self.call_parameters(inputs, training)
        var = tf.exp(logvar)

        if outputs is not None:
            # Sample was given. We calc probs given these values
            sample = outputs
        else:
            sample = self._sample_fn(mu, logvar)
        
        # Metrics for loss
        logprob = self.logprob(sample, mu, var),
        prob = tf.exp(logprob)

        return sample, logprob, prob, mu, logvar

    @staticmethod
    @tf.function
    def _log_normal(x, mu, var, eps=0.0, axis=-1):

        if eps > 0.0:
            var = tf.add(var, eps, name='clipped_var')

        logprob = -0.5 * tf.reduce_sum(
                tf.math.log(2 * tf.cast(np.pi, x.dtype)) 
                + tf.math.log(var) + tf.square(x - mu) / var
            , axis
        )
        #logprob = tf.clip_by_value(logprob, np.log(1e-32), np.log(1-1e-32))
        return logprob

    @tf.function
    def logprob(self, x, mu, var, axis=-1):
        return self._log_normal(x, mu, var, self.epsilon, axis)

    @tf.function 
    def prob(self, x, mu, logvar, axis=-1):
        return tf.exp(self.logprob(x, mu, var, axis))

    @tf.function
    def entropy(self, inputs, inputs_x, mu_x, var_x, training=False, 
        eps=0.0, eps_x=0.0, axis=-1):
        """Compare mu and var against """
        mu = self.mu(inputs, training)
        logvar = self.logvar(inputs, training)
        #mu_logvar = self.mu_logvar(inputs, training)
        #mu, logvar = tf.split(mu_logvar, 2, axis=-1)
        var = tf.math.exp(logvar)  

        entropy = (self.log_normal(inputs_x, mu, var, eps) 
            - self.log_normal(inputs_x, mu_x, var_x, eps_x))

        return entropy



def lognormal_pdf(x, mu, logvar, eps=0.0, axis=-1, clip=1e-18):

    var = tf.math.log(logvar)
    if eps > 0.0:
        var = tf.add(var, eps)

    logprob = - 0.5 * tf.reduce_sum(   
        ( tf.cast(tf.log(2 * np.pi), x.dtype)
         + tf.math.log(var) + tf.square(x - mu) / var)
        ,axis=axis
    )
    #logprob = tf.compat.v2.clip_by_value(logprob, np.log(clip), np.log(1-clip))
    return logprob


def lognormal_kl(x, mu_x, mu_y, logvar_x, logvar_y, 
    eps_x=0.0, eps_y=0.0, axis=-1, clip=1e-24):

    """
    var_x = tf.math.log(logvar_x)
    if eps_x > 0.0:
        var_x = tf.add(var_x, eps_x)

    var_y = tf.math.log(logvar_y)
    if eps_y > 0.0:
        var_y = tf.add(var_y, eps_y)

    entropy = 0.5 * tf.reduce_sum(
        tf.log(var_x) - tf.log(var_y) 
        + var_x / var_y 
        + tf.square(mu_x - mu_y) / var_y - 1
        , axis=-1
    )
    """
    entropy = tf.math.subtract(
        lognormal_pdf(x, mu_y, logvar_y, eps_y)
        ,lognormal_pdf(x, mu_x, logvar_x, eps_x)
    )

    return entropy
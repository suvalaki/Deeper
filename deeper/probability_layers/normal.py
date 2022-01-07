from __future__ import annotations
import tensorflow as tf
import numpy as np
from dataclasses import dataclass, asdict
from collections import namedtuple
from typing import Optional

from deeper.utils.scope import Scope
from deeper.layers.encoder import Encoder, BaseEncoderConfig

tfk = tf.keras
Layer = tfk.layers.Layer
Tensor = tf.Tensor

coalesce = lambda w, num: tf.where(tf.math.is_nan(w), tf.ones_like(w) * num, w)


class RandomNormalEncoder(Layer, Scope):
    @dataclass
    class Config(BaseEncoderConfig):
        embedding_activations: tf.keras.layers.Activation = tf.nn.relu
        epsilon: float = 0.0
        embedding_mu_kernel_initializer: tf.keras.initializers.Initializer = (
            tf.initializers.glorot_normal()
        )
        embedding_mu_bias_initializer: tf.keras.initializers.Initializer = (
            tf.initializers.zeros()
        )
        latent_mu_kernel_initialiazer: tf.keras.initializers.Initializer = (
            tf.initializers.glorot_normal()
        )
        latent_mu_bias_initializer: tf.keras.initializers.Initializer = (
            tf.initializers.zeros()
        )
        embedding_var_kernel_initializer: tf.keras.initializers.Initializer = (
            tf.initializers.glorot_normal()
        )
        embedding_var_bias_initializer: tf.keras.initializers.Initializer = (
            tf.initializers.zeros()
        )
        latent_var_kernel_initialiazer: tf.keras.initializers.Initializer = (
            tf.initializers.glorot_normal()
        )
        latent_var_bias_initializer: tf.keras.initializers.Initializer = (
            tf.initializers.ones()
        )
        fixed_mu: Optional[float] = None
        fixed_var: Optional[float] = None
        connected_weights: bool = True
        embedding_mu_dropout: float = 0.0
        embedding_var_dropout: float = 0.0

    Output = namedtuple(
        "RandomNormalEncoderOutput",
        ["sample", "logprob", "prob", "mu", "logvar", "var"],
    )

    @classmethod
    def from_config(cls, config: RandomNormalEncoder.Config, **kwargs):
        return cls(**asdict(config), **kwargs)

    def __init__(
        self,
        latent_dimension,
        embedding_dimensions,
        embedding_activations=tf.nn.relu,
        var_scope="normal_encoder",
        bn_before=False,
        bn_after=False,
        epsilon=0.0,
        embedding_mu_kernel_initializer=tf.initializers.glorot_normal(),
        embedding_mu_bias_initializer=tf.initializers.zeros(),
        latent_mu_kernel_initialiazer=tf.initializers.glorot_normal(),
        latent_mu_bias_initializer=tf.initializers.zeros(),
        embedding_var_kernel_initializer=tf.initializers.glorot_normal(),
        embedding_var_bias_initializer=tf.initializers.zeros(),
        latent_var_kernel_initialiazer=tf.initializers.glorot_normal(),
        latent_var_bias_initializer=tf.initializers.ones(),
        fixed_mu=None,
        fixed_var=None,
        connected_weights=True,
        embedding_mu_dropout=0.0,
        embedding_var_dropout=0.0,
    ):
        """Probability Layer multivariate random normal

        Parameters:
        -----------
        latent_dimensions: int, Output dimension of the probabilistic layer
        embedding_dimensions: list, the dimension of each layer from input to
            output for the embedding layers of the encoder for mu and logvar
        embedding_activation = the tensorflow activation function to apply to
            each layer of the embedding encoder for mu and logvar
        bn_before: bool, flag whether to apply batch normalisation before
            activation in the encoder for mu and logvar
        bn_after: bool, glag whether to apply batch normalisation after
            activation in the encoder for mu and logvar
        fixed_mu: value (to be implemented) A fixed value for mu
        fixed_var: value (to be implemented) A fixed value for var
        connected_weights: bool, whether to train mu and var as a fully
            connected network.

        """
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
                var_scope=self.v_name("mu_encoder"),
                bn_before=self.bn_before,
                bn_after=self.bn_after,
                embedding_kernel_initializer=embedding_mu_kernel_initializer,
                embedding_bias_initializer=embedding_mu_bias_initializer,
                latent_kernel_initialiazer=latent_mu_kernel_initialiazer,
                latent_bias_initializer=latent_mu_bias_initializer,
                embedding_dropout=embedding_mu_dropout,
            )
            self.logvar = Encoder(
                latent_dim=self.latent_dimension,
                embedding_dimensions=self.embedding_dimensions,
                activation=self.embedding_activation,
                var_scope=self.v_name("logvar_encoder"),
                bn_before=self.bn_before,
                bn_after=self.bn_after,
                embedding_kernel_initializer=embedding_var_kernel_initializer,
                embedding_bias_initializer=embedding_var_bias_initializer,
                latent_kernel_initialiazer=latent_var_kernel_initialiazer,
                latent_bias_initializer=latent_var_bias_initializer,
                embedding_dropout=embedding_var_dropout,
            )
        else:
            self.mu_logvar = Encoder(
                latent_dim=2 * self.latent_dimension,
                embedding_dimensions=[x for x in self.embedding_dimensions],
                activation=self.embedding_activation,
                var_scope=self.v_name("mu_encoder"),
                bn_before=self.bn_before,
                bn_after=self.bn_after,
                embedding_kernel_initializer=embedding_mu_kernel_initializer,
                embedding_bias_initializer=embedding_mu_bias_initializer,
                latent_kernel_initialiazer=latent_mu_kernel_initialiazer,
                latent_bias_initializer=latent_mu_bias_initializer,
                embedding_dropout=embedding_mu_dropout,
            )

    @tf.function
    def call_parameters(
        self,
        inputs,
        training=False,
        apply_epsilon=True,
    ):
        """Return the parameters for this normal distribution

        Parameters:
        -----------
        inputs: Input vector to the Encoder`
        training: bool flag for whether the layer is training
        apply_epsilon: bool flag for whether toa djust the logvar by epsilon

        """

        if not self.connected_weights:
            mu = self.mu(
                inputs,
                training,
            )
            logvar = self.logvar(
                inputs,
                training,
            )
        else:
            mu_logvar = self.mu_logvar(
                inputs,
                training,
            )
            mu, logvar = tf.split(mu_logvar, 2, axis=-1)

        if apply_epsilon:
            var = tf.add(tf.exp(logvar), self.epsilon)
            logvar = tf.math.log(var)

        mu = tf.identity(mu, name=self.v_name("mu"))
        logvar = tf.identity(logvar, name=self.v_name("logvar"))
        var = tf.exp(logvar, name=self.v_name("var"))

        if self.fixed_var is not None:
            var = tf.cast(tf.fill(tf.shape(mu), self.fixed_var), inputs.dtype)
            logvar = tf.math.log(var)

        # Convert nans to numbers
        mu = coalesce(mu, 0.0)
        logvar = coalesce(logvar, 0.0)
        var = coalesce(var, 1.0)

        return mu, logvar, var

    @tf.function
    def call_mu(
        self,
        inputs,
        training=False,
        apply_epsilon=True,
    ):
        """Get the mean of the network given the inputs

        Parameters
        ----------
        inputs: Input vector to the Encoder`
        training: bool flag for whether the layer is training
        apply_epsilon: bool flag for whether toa djust the logvar by epsilon
        ldr: allowable lower depth range. call through embedding layers will
            include layers with a lesser than or equal to index value to this.
            The embedding layers between aldr and audr will be skipped.
        audr: allowable upper depth range. call through embedding layers will
            include layers with a higher than or equal to index value to this.
        """
        mu, logvar, var = self.call_parameters(
            inputs,
            training,
        )
        return mu

    @tf.function
    def call_logvar(
        self,
        inputs,
        training=False,
        apply_epsilon=True,
    ):
        """Get the logvar of the network given the inputs

        Parameters
        ----------
        inputs: Input vector to the Encoder`
        training: bool flag for whether the layer is training
        apply_epsilon: bool flag for whether toa djust the logvar by epsilon
        ldr: allowable lower depth range. call through embedding layers will
            include layers with a lesser than or equal to index value to this.
            The embedding layers between aldr and audr will be skipped.
        audr: allowable upper depth range. call through embedding layers will
            include layers with a higher than or equal to index value to this.
        """
        mu, logvar, var = self.call_parameters(
            inputs,
            training,
            apply_epsilon,
        )
        return logvar

    @tf.function
    def call_var(
        self,
        inputs,
        training=False,
        apply_epsilon=True,
    ):
        """Get the var of the network given the inputs

        Parameters
        ----------
        inputs: Input vector to the Encoder`
        training: bool flag for whether the layer is training
        apply_epsilon: bool flag for whether toa djust the logvar by epsilon
        ldr: allowable lower depth range. call through embedding layers will
            include layers with a lesser than or equal to index value to this.
            The embedding layers between aldr and audr will be skipped.
        audr: allowable upper depth range. call through embedding layers will
            include layers with a higher than or equal to index value to this.
        """
        mu, logvar, var = self.call_parameters(
            inputs,
            training,
            apply_epsilon,
        )
        var = tf.exp(logvar)
        return var

    @staticmethod
    @tf.function
    def _sample_fn(mu, var):
        """Reparameterisation trick sample from a random normal distribution
        with the given mean (mu) and variance (var)

        Paramters:
        ----------
        mu: tf.tensor mean of the distribution
        var: tf.tensor, variance of the distribution
        name: name of output sample tensor
        """
        sample_shape = tf.shape(mu)
        r_norm = tf.cast(
            tf.random.normal(sample_shape, mean=0.0, stddev=1.0), mu.dtype
        )
        sample = tf.add(mu, tf.multiply(r_norm, tf.sqrt(var)))
        return sample

    @tf.function
    def sample(
        self,
        inputs,
        training=False,
        apply_epsilon=False,
        name="sample",
    ):
        """Sample from this distribution layer.

        Parameters:
        -----------
        inputs: Input vector to the
        training: bool flag for whether the layer is training
        apply_epsilon: bool flag for whether toa djust the logvar by epsilon
        ldr: allowable lower depth range. call through embedding layers will
            include layers with a lesser than or equal to index value to this.
            The embedding layers between aldr and audr will be skipped.
        audr: allowable upper depth range. call through embedding layers will
            include layers with a higher than or equal to index value to this.
        name: name of output sample tensor
        """

        mu, logvar, var = self.call_parameters(
            inputs,
            training,
            apply_epsilon,
        )
        sample = self._sample_fn(mu, var)
        return sample

    @tf.function
    def call(
        self,
        inputs,
        training=False,
        outputs=None,
        apply_epsilon=True,
    ):

        mu, logvar, var = self.call_parameters(
            inputs,
            training,
            apply_epsilon,
        )

        if outputs is not None:
            # Sample was given. We calc probs given these values
            sample = outputs
        else:
            sample = self._sample_fn(mu, var)

        # Metrics for loss
        logprob = self.logprob(sample, mu, var)
        prob = tf.exp(logprob)
        return self.Output(sample, logprob, prob, mu, logvar, var)

    @staticmethod
    # @tf.function
    def _log_normal(x, mu, var, eps=0.0, axis=-1):

        if eps > 0.0:
            var = tf.add(var, eps, name="clipped_var")

        # kernel = - 0.5 * (
        #    tf.math.log(2 * tf.cast(np.pi, mu.dtype))
        #    + tf.math.log(var) + tf.square(x - mu) / var
        # )

        kernel = -0.5 * (
            tf.math.log(2 * tf.cast(np.pi, x.dtype))
            + tf.math.log(var)
            + tf.math.divide_no_nan(tf.square(x - mu), var)
        )

        logprob = tf.reduce_sum(kernel, axis)
        return logprob

    @tf.function
    def logprob(self, x, mu, var, axis=-1):
        return coalesce(self._log_normal(x, mu, var, 0.0, axis), 0.0)

    @tf.function
    def prob(self, x, mu, logvar, axis=-1):
        return coalesce(tf.exp(self.logprob(x, mu, var, axis)), 0.0)


def lognormal_pdf(x, mu, logvar, eps=0.0, axis=-1, clip=1e-18):

    eps = tf.cast(eps, dtype=x.dtype)
    mu = tf.cast(mu, dtype=x.dtype)
    logvar = tf.cast(logvar, dtype=x.dtype)
    var = tf.math.exp(logvar)

    var = tf.cond(eps > 0.0, lambda: tf.add(var, eps), lambda: var)

    logprob = -0.5 * (
        tf.cast(tf.shape(x)[axis], x.dtype)
        * (logvar + tf.cast(tf.math.log(2 * np.pi), x.dtype))
        + tf.reduce_sum(tf.square(x - mu) / tf.math.sqrt(var), axis)
    )

    # logprob = tf.compat.v2.clip_by_value(logprob, np.log(clip), np.log(1-clip))
    # logprob_no_nan = tf.where(
    #     tf.math.is_nan(logprob), tf.zeros_like(logprob), logprob
    # )
    return logprob


def lognormal_kl(
    x, mu_x, mu_y, logvar_x, logvar_y, eps_x=0.0, eps_y=0.0, axis=-1, clip=0.0
):

    var_x = tf.math.exp(logvar_x)
    if eps_x > 0.0:
        var_x = tf.add(var_x, eps_x)

    var_y = tf.math.exp(logvar_y)
    if eps_y > 0.0:
        var_y = tf.add(var_y, eps_y)

    w = (
        logvar_x
        - logvar_y
        + var_x / var_y
        + tf.square(mu_x - mu_y) / var_y
        - 1
    )

    # w_no_nan = tf.where(tf.math.is_nan(w), tf.zeros_like(w), w)

    # entropy = 0.5 * tf.reduce_sum(
    #     w_no_nan,
    #     axis=-1,
    # )

    entropy = 0.5 * tf.reduce_sum(w, axis=-1)

    return entropy

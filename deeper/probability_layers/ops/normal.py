import tensorflow as tf
import numpy as np
from typing import Optional

tfk = tf.keras
Layer = tfk.layers.Layer
Tensor = tf.Tensor


def logvar_computation(logvar: Tensor, epsilon: float = 0.0) -> Tensor:
    """A helper function to ensure function utilising a switch between 
    var and logvar are functional. 
    """

    var = tf.math.exp(logvar)

    if epsilon > 0.0:
        var = tf.math.add(var, epsilon)

    return var


def lognormal_pdf(
    x: Tensor, mu: Tensor, logvar: Tensor, epsilon: float = 0.0
) -> Tensor:
    """Calculate the lognormal_pdf (along each axis) for the input tensor x
    with mean mu and variance var (exp(logvar)) adjusted by epsilon. 

    Input
    -----
    x: Tensor: values at which to calculate the lognormal pdf 
    mu: Tensor: mean values for the distribution 
    var: (optional) Tensor: variance values for the distribution. Must be 
        included if logvar is None
    logvar: (optional) Tensor: logvariance values for the distribution. Must be
        included if var is None 
    epsilon: float: minimum value for the variance. 

    Output: Tensor of marginal probabilities. The shape of which is the same as 
        the input x
    """
    var = logvar_computation(logvar, epsilon)
    logprob = -0.5 * (
        tf.math.log(2.0 * tf.cast(np.pi, x.dtype))
        + tf.math.log(var)
        + tf.math.square(x - mu) / var
    )
    return logprob


def mv_lognormal_pdf(
    x: Tensor, mu: Tensor, logvar: Tensor, epsilon: float = 0.0, axis: int = -1
) -> Tensor:
    """Calculate the lognormal_pdf (along each axis) for the input tensor x
    with mean mu and variance var (exp(logvar)) adjusted by epsilon. We assume
    the covariance matrix is zeros everywhere except on the diagonals (whos 
    values are specified by var/logvar). As such each dimension underlying the
    distribution is independent of the others.

    Input
    -----
    x: Tensor: values at which to calculate the lognormal pdf 
    mu: Tensor: mean values for the distribution 
    var: (optional) Tensor: variance values for the distribution. Must be 
        included if logvar is None
    logvar: (optional) Tensor: logvariance values for the distribution. Must be
        included if var is None 
    epsilon: float: minimum value for the variance. 
    axis: the axis uppon which to calculate the joint probability 
        (via logsum of dimensions)

    Output: Tensor of joint probabilities Output shape is a scalar.

    """
    comp_logprob = lognormal_pdf(x, mu, logvar, epsilon)
    logprob = tf.reduce_sum(comp_logprob, axis=axis)

    return logprob


def normal_sample(
    mu: Tensor,
    logvar: Tensor,
    epsilon: float = 0.0,
    name: Optional[str] = None,
) -> Tensor:
    """Create a sample from a normal distribution with mean and var via
    the reparameterisation trick (enabling gradient flow)

    Input
    -----
    mu: Tensor: 
    var: Tensor: 
    logvar: Tensor:
    epsilon: float
    name: str 

    Output: 
    """

    var = logvar_computation(logvar, epsilon)
    std_normal_sample = tf.random.normal(tf.shape(mu))

    # Reparameterisation trick
    sample = mu + std_normal_sample * tf.math.sqrt(var)

    if name is not None:
        sample = tf.identity(sample, name=name)

    return sample


def std_normal_kl_divergence(
    mu: Tensor,
    logvar: Tensor,
    epsilon: float = 0.0,
    axis: int = -1,
    name: Optional[str] = None,
) -> Tensor:
    """Simplified version of kl-divergence for a proposal distribution 
    following a normal distribution and a prior distribution following a std
    MVN(0,I). 

    Input
    -----
    mu: Tensor: mean values for the proposal distribution 
    var: (optional) Tensor: variance values for the proposal distribution. Must 
        be included if logvar is None
    logvar: (optional) Tensor: logvariance values for the proposal distribution. 
        Must be included if var is None 
    epsilon: float: minimum value for the variance. 
    axis: the axis uppon which to calculate the joint probability 
        (via logsum of dimensions)
    """
    var = logvar_computation(logvar, epsilon)
    kl_divergence = 0.5 * tf.reduce_sum(
        1 + tf.math.log(var) - tf.math.square(mu) - var, axis=axis
    )
    if name is not None:
        kl_divergence = tf.identity(kl_divergence, name=name)

    return kl_divergence

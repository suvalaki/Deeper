import tensorflow as tf
import numpy as np
from typing import Optional


Tensor = tf.Tensor


def logprob_divergence(px, qx, axis=-1, name="logprob_divergence"):
    d = tf.reduce_sum(tf.math.subtract(tf.log(px), tf.log(qx)), axis=axis, name=name)
    return d


def kl_divergence(px, qx, axis=-1, name="kl_divergence"):
    """
    dkl(P|Q) = E_P [ log(P) - log(Q) ]
             = integrate_(x) P * (log(P) - log(Q))

    """
    dkl = tf.reduce_sum(
        tf.math.multiply(px, tf.math.subtract(tf.log(px), tf.log(qx))),
        axis=axis,
        name=name,
    )
    return dkl


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
    kl_divergence = 0.5 * tf.reduce_sum(1 + tf.math.log(var) - tf.math.square(mu) - var, axis=axis)
    if name is not None:
        kl_divergence = tf.identity(kl_divergence, name=name)

    return kl_divergence

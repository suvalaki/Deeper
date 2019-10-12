import tensorflow as tf 
import numpy as np 


def logprob_divergence(px, qx, axis=-1, name='logprob_divergence'):
    d = tf.reduce_sum(
        tf.math.subtract(tf.log(px), tf.log(qx)), 
        axis=axis, 
        name=name
    )
    return d


def kl_divergence(px, qx, axis=-1, name='kl_divergence'):
    """
    dkl(P|Q) = E_P [ log(P) - log(Q) ]
             = integrate_(x) P * (log(P) - log(Q))

    """
    dkl = tf.reduce_sum(
        tf.math.multiply(px, tf.math.subtract(tf.log(px), tf.log(qx))), 
        axis=axis, 
        name=name
    )
    return dkl


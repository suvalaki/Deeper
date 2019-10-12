import tensorflow as tf 
import numpy as np 

def kl_divergence(px, qx, axis=-1, name='kl_divergence'):
    dkl = tf.reduce_sum(px * (tf.log(px) - tf.log(qx)), axis=axis, name=name)
    return dkl


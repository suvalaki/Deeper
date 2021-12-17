import tensorflow as tf


def tile_scalar_value(x, shape):
    """Replicates the scalar value over a tensor of given shape"""
    return tf.ones(shape) * x


def tile_scalar_value_for_concat(x, y):
    """Expand the dimensions on a scalar value so that it can be
    concat on the last dimension of another tensor"""
    return tile_scalar_value(x, tf.stack([tf.shape(y)[:-1], [1]], axis=-1)[0])

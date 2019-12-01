import unittest
import tensorflow as tf
import numpy as np
from deeper.layers.categorical import CategoricalEncoder

tf.enable_eager_execution()


encoder = CategoricalEncoder(
    latent_dimension=10,
    embedding_dimensions=[15, 15],
    embedding_activation=tf.nn.tanh,
)


z = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]).astype(
    float
)
res = encoder(z)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
encoder.entropy(z, res[1] > 0)

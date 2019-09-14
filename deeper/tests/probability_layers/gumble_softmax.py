import unittest
import numpy as np
import tensorflow as tf
from deeper.probability_layers.gumble_softmax import GumbleSoftmaxLayer 

tf.enable_eager_execution()

layer = GumbleSoftmaxLayer()

z = np.array([1,2,3,4]).astype(float)
layer(z, 1.0, 10)
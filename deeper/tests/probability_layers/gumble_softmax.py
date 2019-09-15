import unittest
import numpy as np
import tensorflow as tf
from deeper.probability_layers.gumble_softmax import GumbleSoftmaxLayer 

tf.enable_eager_execution()

layer = GumbleSoftmaxLayer()

z = np.array([[10000,2,3,4, 0.00001]]).astype(float)

z = np.array([[-0.03840736,  2.58452828]])

layer(z, 1.0 )
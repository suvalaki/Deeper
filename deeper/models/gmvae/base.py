import tensorflow as tf
from typing import NamedTuple



class GmvaeNetBase(tf.keras.layers.Layer):

    class Output(NamedTuple):
        ...
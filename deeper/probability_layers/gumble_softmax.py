import tensorflow as tf
from deeper.utils.scope import Scope

tfk = tf.keras

Layer = tfk.layers.Layer


class GumbleSoftmaxLayer(Layer, Scope):
    def __init__(self, var_scope="GumbleSoftmax", axis=-1):

        Layer.__init__(self)
        Scope.__init__(self, var_scope)
        self.axis = axis

    @tf.function
    def _inverse_gumble_transformation(self, x):
        return -tf.math.log(-tf.math.log(x))

    @tf.function
    def _gumble_softmax(self, logits, gumble_sample, temperature):
        adjusted_logits = (logits + gumble_sample) / temperature
        adjusted_softmax = tf.nn.softmax(adjusted_logits)
        return adjusted_softmax

    @tf.function
    def _sample_one(self, logits, temperature=None):
        uniform_sample = tf.random.uniform(
            shape=tf.shape(logits),
            dtype=self.dtype,
            name="random_uniform_sample",
        )
        gumble_sample = self._inverse_gumble_transformation(uniform_sample)
        softmax_trick = self._gumble_softmax(
            logits, gumble_sample, temperature
        )
        return softmax_trick

    @tf.function
    def call(self, logits, temperature=None, samples=1):
        with tf.name_scope(self.name):
            if samples > 1:
                output = tf.stack(
                    [
                        self._sample_one(logits, temperature)
                        for i in range(samples)
                    ],
                    axis=self.axis,
                )
            else:
                output = self._sample_one(logits, temperature)
        return output

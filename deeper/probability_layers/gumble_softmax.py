import tensorflow as tf

tfk = tf.keras

class GumbleSoftmaxLayer(tfk.layers.Layer):
    def __init__(
        self,
        dtype=tf.float32,
        name='GumbleSoftmax',
        axis=-1
    ):
        self.dtype=dtype
        self.axis=axis

    @tf.function
    @staticmethod
    def _inverse_gumble_transformation(x):
        return - tf.math.log(-tf.math.log(x))
    
    @tf.function
    @staticmethod
    def _gumble_softmax(logits, gumble_sample, temperature):
        adjusted_logits = (tf.math.log(logits) + gumple_sample)/temperature
        adjusted_softmax = tf.nn.softmax(adjusted_logits)
        return adjusted_softmax

    @tf.function
    def _sample_one(self, logits, temperature=None):
        uniform_sample = tf.random.uniform(
            shape=tf.shape(logits),
            dtype=self.dtype,
            name='random_uniform_sample'
        )
        gumble_sample = self._inverse_gumble_transformation(uniform_sample)
        softmax_trick = self._gumble_softmax(logits, gumble_sample, temperature)
        return softmax_trick

    @tf.function
    def call(self, logits, temperature=None, samples=1):
        with tf.name_scope(self.name):
            if sample > 1:
                output = tf.stack([
                    self._sample_one(logits, temperature) 
                    for i in range(samples)
                    ],
                    axis=self.axis
                ) 
            else:
                output = self._sample_one(logits, temperature)
        return output


from __future__ import annotations
import tensorflow as tf
import numpy as np

from typing import Optional

from deeper.layers.encoder import Encoder
from deeper.utils.scope import Scope
from pydantic import Field

tfk = tf.keras

Layer = tfk.layers.Layer


class SigmoidEncoder(Encoder):
    class Config(Encoder.Config):
        latent_dim: int = Field(1, const=True)
        epsilon: float = 0.0

    class Output(tf.experimental.ExtensionType):
        logits: tf.Tensor
        logprob: tf.Tensor
        prob: tf.Tensor
        entropy: Optional[tf.Tensor] = None

    def __init__(self, config: SigmoidEncoder.Config, **kwargs):
        self.epsilon = config.epsilon
        super().__init__(
            **{k: v for k, v in dict(config).items() if k in Encoder.Config.__fields__.keys()},
            **kwargs
        )

    @tf.function
    def call_logits(self, inputs, training=False):
        inputs = tf.cast(inputs, self.dtype)
        logits = super().call(inputs, training)
        if self.epsilon > 0.0:
            maxval = np.log(1.0 - self.epsilon) - np.log(self.epsilon)
            logits = tf.compat.v2.clip_by_value(logits, -maxval, maxval, "clipped")
        return logits

    @tf.function
    def _prob(self, logits):
        prob = tf.nn.sigmoid(logits, name="probs")
        return prob

    @tf.function
    def prob(self, inputs, training=False):
        inputs = tf.cast(inputs, self.dtype)
        logits, probs = self.call(inputs, training)
        return probs

    @tf.function
    def entropy(self, x, y, training=False):
        x = tf.cast(x, self.dtype)
        if y is not None:
            y = tf.cast(y, self.dtype)
        logits = self.call_logits(x, training)
        ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits, name="entropy")
        return ent

    @tf.function
    def call(self, inputs, training=False, y=None):
        inputs = tf.cast(inputs, self.dtype)
        if y is not None:
            y = tf.cast(y, self.dtype)
        logits = self.call_logits(inputs, training)
        probs = self._prob(logits)
        if y is not None:
            ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits, name="entropy")
        else:
            ent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=probs, logits=logits, name="entropy"
            )
        logprob = -ent
        prob = probs

        return SigmoidEncoder.Output(logits=logits, logprob=logprob, prob=prob, entropy=ent)

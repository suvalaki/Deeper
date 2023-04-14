from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Activation


from typing import Tuple, Union, Optional, Sequence, NamedTuple
from dataclasses import dataclass, asdict

from deeper.layers.encoder import Encoder
from deeper.probability_layers.normal import RandomNormalEncoder
from deeper.utils.function_helpers.decorators import inits_args
from deeper.probability_layers.ops.normal import std_normal_kl_divergence
from deeper.utils.tf.experimental.extension_type import ExtensionTypeIterableMixin


class VaeLossNetLatent(tf.keras.layers.Layer):
    class Input(tf.experimental.ExtensionType, ExtensionTypeIterableMixin):
        mu: tf.Tensor
        logvar: tf.Tensor

        @classmethod
        def from_VaeEncoderNet(cls, x: VaeEncoderNet) -> VaeLossNetLatent.Input:
            return VaeLossNetLatent.Input(x.mu, x.logvar)

    def __init__(self, latent_eps=0.0, name="latent_kl", **kwargs):
        super(VaeLossNetLatent, self).__init__(name=name, **kwargs)
        self.latent_eps = latent_eps

    def latent_kl(self, mu, logvar, training=False):
        kl = std_normal_kl_divergence(mu, logvar, epsilon=self.latent_eps)
        self.add_metric(kl, name=self.name)
        return kl

    def call(self, x: VaeLossNetLatent.Input, training=False) -> tf.Tensor:
        return self.latent_kl(x.mu, x.logvar, training)

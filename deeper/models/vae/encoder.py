from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Activation


from typing import Tuple, Union, Optional, Sequence
from dataclasses import dataclass, asdict

from deeper.layers.encoder import Encoder
from deeper.probability_layers.normal import (
    RandomNormalEncoder,
)
from deeper.utils.function_helpers.decorators import inits_args


class VaeEncoderNet(RandomNormalEncoder):

    Config = RandomNormalEncoder.Config
    Output = RandomNormalEncoder.Output

    @classmethod
    def from_config(cls, config: VaeEncoderNet.Config, **kwargs):
        kw_ = {
            k: v
            for k, v in asdict(config).items()
            if k not in ["latent_dim", "embedding_dimensions"]
        }
        return cls(
            config.latent_dim, config.embedding_dimensions, **kw_, **kwargs
        )

    def __init__(self, latent_dimension, embedding_dimensions, **kwargs):
        super().__init__(latent_dimension, embedding_dimensions, **kwargs)

    @tf.function
    def call(self, x, training=False) -> VaeNet.VaeNetOutput:
        return self.Output(*super().call(x, training))

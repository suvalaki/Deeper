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
    class Config(RandomNormalEncoder.Config):
        pass

    Output = RandomNormalEncoder.Output

    def __init__(self, config: VaeEncoderNet.Config, **kwargs):
        super().__init__(config, **kwargs)

    @tf.function
    def call(self, x, training=False) -> VaeNet.VaeNetOutput:
        return self.Output(*super().call(x, training))

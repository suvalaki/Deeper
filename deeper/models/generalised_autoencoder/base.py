from __future__ import annotations

import tensorflow as tf

from pydantic import BaseModel, Field
from typing import Sequence
from abc import ABC, abstractmethod


class LatentParser(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_pred, training=False):
        ...


class AutoencoderBase(ABC, tf.keras.layers.Layer):
    class Config(BaseModel):

        input_dimensions: MultipleObjectiveDimensions = Field()
        output_dimensions: MultipleObjectiveDimensions = Field()
        encoder_embedding_dimensions: Sequence[int] = Field()
        decoder_embedding_dimensions: Sequence[int] = Field()
        latent_dim: int = Field()

        embedding_activations: tf.keras.layers.Layer = tf.keras.layers.ReLU()
        bn_before: bool = False
        bn_after: bool = False

        class Config:
            arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def split_outputs(self, y) -> SplitCovariates:
        ...

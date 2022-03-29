from __future__ import annotations

import tensorflow as tf

from pydantic import BaseModel, Field
from typing import Sequence
from abc import ABC, abstractmethod

from deeper.utils.model_mixins import LatentMixin, ReconstructionMixin

from deeper.utils.type_getter import NetworkTypeGetterBase


class AutoencoderTypeGetterBase(NetworkTypeGetterBase):
    @abstractmethod
    def get_latent_parser_type(self):
        ...


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

        embedding_activations: tf.keras.layers.Activation = (
            tf.keras.layers.Activation("relu")
        )
        bn_before: bool = False
        bn_after: bool = False

        class Config:
            arbitrary_types_allowed = True
            smart_union = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def split_outputs(self, y) -> SplitCovariates:
        ...


class AutoencoderModelBaseMixin(LatentMixin, ReconstructionMixin):
    def __init__(
        self,
        weight_getter,
        network,
        latent_parser,
        reconstruction_parser,
        **kwargs
    ):
        LatentMixin.__init__(self, weight_getter, network, latent_parser)
        ReconstructionMixin.__init__(
            self, weight_getter, network, reconstruction_parser
        )

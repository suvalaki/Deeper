from __future__ import annotations

import tensorflow as tf

from pydantic import BaseModel, Field
from typing import Sequence, Union, Tuple
from abc import ABC, abstractmethod

from deeper.utils.model_mixins import LatentMixin, ReconstructionMixin
from deeper.utils.type_getter import NetworkTypeGetterBase
from deeper.optimizers.automl.tunable_types import (
    TunableModelMixin,
    TunableActivation,
    TunableBoolean,
)
from deeper.models.generalised_autoencoder.tunable import (
    TunableLatentDimensions,
    TunableEmbeddingDimensions,
    TunableDecodingDimensionsReflectReverse,
)


class MultipleObjectiveDimensions(BaseModel):
    regression: int = Field()
    boolean: int = Field()
    ordinal: Sequence[int] = Field()
    categorical: Sequence[int] = Field()

    def as_list(self):
        return [self.regression, self.boolean, self.ordinal, self.categorical]

    @classmethod
    def as_null(cls):
        return cls(regression=0, boolean=0, ordinal=(0,), categorical=(0,))


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
    class Config(TunableModelMixin):

        input_dimensions: MultipleObjectiveDimensions = Field()
        output_dimensions: MultipleObjectiveDimensions = Field()

        # By leaving these blank on initialisation we allow for hyper tuning with defaults
        encoder_embedding_dimensions: Tuple[int, ...] = TunableEmbeddingDimensions()
        decoder_embedding_dimensions: Tuple[int, ...] = TunableDecodingDimensionsReflectReverse()
        latent_dim: int = TunableLatentDimensions(10)

        embedding_activations: Union[tf.keras.layers.Activation] = TunableActivation("relu")
        bn_before: bool = False
        bn_after: bool = False

        encoder_kwargs: dict = dict()
        decoder_kwargs: dict = dict()

        class Config:
            arbitrary_types_allowed = True
            smart_union = True

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)

            # Update backreferences for tunable parameters
            if isinstance(self.latent_dim, TunableLatentDimensions):
                self.latent_dim.update_backref(self)

            if isinstance(self.encoder_embedding_dimensions, TunableEmbeddingDimensions):
                self.encoder_embedding_dimensions.update_backref(self)

            if isinstance(self.decoder_embedding_dimensions, TunableEmbeddingDimensions):
                self.decoder_embedding_dimensions.update_backref(self)

    Config.update_forward_refs()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def split_outputs(self, y) -> SplitCovariates:
        ...


class AutoencoderModelBaseMixin(LatentMixin, ReconstructionMixin):
    def __init__(self, weight_getter, network, latent_parser, reconstruction_parser, **kwargs):
        LatentMixin.__init__(self, weight_getter, network, latent_parser)
        ReconstructionMixin.__init__(self, weight_getter, network, reconstruction_parser)

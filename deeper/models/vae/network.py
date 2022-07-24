from __future__ import annotations
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Activation

from typing import Tuple, Union, Optional, Sequence, NamedTuple

from deeper.layers.encoder import Encoder
from deeper.probability_layers.normal import RandomNormalEncoder
from deeper.probability_layers.ops.normal import std_normal_kl_divergence
from deeper.layers.data_splitter import split_inputs, unpack_dimensions
from deeper.utils.function_helpers.decorators import inits_args
from deeper.models.vae.encoder import VaeEncoderNet
from deeper.models.vae.decoder import VaeReconstructionNet
from deeper.models.vae.utils import VaeTypeGetter
from deeper.utils.tf.experimental.extension_type import ExtensionTypeIterableMixin

from deeper.models.generalised_autoencoder.base import MultipleObjectiveDimensions

from deeper.probability_layers.normal import (
    lognormal_kl,
    lognormal_pdf,
)
from tensorflow.python.keras.metrics import categorical_accuracy, accuracy

from deeper.utils.tf.keras.models import Model
from deeper.models.generalised_autoencoder.base import AutoencoderBase
from pydantic import BaseModel, Field


def reduce_groups(fn, x_grouped: Tuple[tf.Tensor, ...]):
    if len(x_grouped) <= 1:
        return fn(x_grouped[0])
    return tf.concat([fn(z) for z in x_grouped], -1)


class VaeNet(AutoencoderBase):
    class Config(VaeTypeGetter, AutoencoderBase.Config):
        class Config:
            arbitrary_types_allowed = True

        @property
        def _ignored_encoder_fields(self):
            return ["latent_dim", "activation", "embedding_activations", "embedding_dimensions"]

        @property
        def _ignored_decoder_fields(self):
            return [
                "output_dimensions",
                "decoder_embedding_dimensions",
                "embedding_activations",
                "embedding_dimensions",
            ]

        def parse_tunable(self, hp, prefix=""):

            # Add the encoder/decoder fields to the model
            base_encoder_fields = VaeEncoderNet.Config().parse_tunable(
                hp, prefix + "encoder_kwargs_"
            )
            base_decoder_fields = VaeReconstructionNet.Config(
                output_dimensions=MultipleObjectiveDimensions.as_null(),
                decoder_embedding_dimensions=[],
            ).parse_tunable(hp, prefix + "decoder_kwargs_")
            self.encoder_kwargs = {
                k: v
                for k, v in dict(base_encoder_fields).items()
                if k not in self._ignored_encoder_fields and k not in self.encoder_kwargs.keys()
            }
            self.decoder_kwargs = {
                k: v
                for k, v in dict(base_decoder_fields).items()
                if k not in self._ignored_decoder_fields and k not in self.decoder_kwargs.keys()
            }
            return super().parse_tunable(hp, prefix)

    class Output(tf.experimental.ExtensionType, ExtensionTypeIterableMixin, VaeTypeGetter):
        # Encoder/Latent variables
        qz_g_x: VaeEncoderNet.Output
        # DecoderVariables
        px_g_z: VaeReconstructionNet.Output

    def __init__(
        self,
        config: VaeNet.Config,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config

        # Input Dimension Calculation
        (
            self.input_ordinal_dimension,
            self.input_ordinal_dimension_tot,
            self.input_categorical_dimension,
            self.input_categorical_dimension_tot,
            self.input_dim,
        ) = unpack_dimensions(*config.input_dimensions.as_list())

        # Encoder
        self.graph_qz_g_x = VaeEncoderNet(
            VaeEncoderNet.Config(
                latent_dim=config.latent_dim,
                activation=config.embedding_activations,
                embedding_activations=config.embedding_activations,
                embedding_dimensions=config.encoder_embedding_dimensions,
                **{
                    k: v
                    for k, v in config.encoder_kwargs.items()
                    if k not in self.config._ignored_encoder_fields
                },
            ),
            **kwargs,
        )

        # Decoder
        self.graph_px_g_z = VaeReconstructionNet(
            VaeReconstructionNet.Config(
                output_dimensions=config.output_dimensions,
                decoder_embedding_dimensions=config.decoder_embedding_dimensions,
                embedding_activations=config.embedding_activations,
                embedding_dimensions=config.decoder_embedding_dimensions,
                **{
                    k: v
                    for k, v in config.decoder_kwargs.items()
                    if k not in self.config._ignored_decoder_fields
                },
            ),
            **kwargs,
        )

    @property
    def encoder(self):
        return self.graph_qz_g_x

    @property
    def decoder(self):
        return self.graph_px_g_z

    @property
    def encoder(self):
        return self.graph_qz_g_x

    @property
    def output_ordinal_dimension(self):
        return self.graph_px_g_z.output_ordinal_dimension

    @property
    def output_ordinal_dimension_tot(self):
        return self.graph_px_g_z.output_ordinal_dimension_tot

    @property
    def output_categorical_dimension(self):
        return self.graph_px_g_z.output_categorical_dimension

    @property
    def output_categorical_dimension_tot(self):
        return self.graph_px_g_z.output_categorical_dimension_tot

    @property
    def output_dim(self):
        return self.graph_px_g_z.output_dim

    @tf.function
    def split_inputs(self, x) -> SplitCovariates:
        return split_inputs(
            x,
            self.input_regression_dimension,
            self.input_boolean_dimension,
            self.input_ordinal_dimension,
            self.input_categorical_dimension,
        )

    @tf.function
    def split_outputs(self, y) -> SplitCovariates:
        return self.graph_px_g_z.splitter(y)

    @tf.function
    def call(self, x, training=False) -> VaeNet.VaeNetOutput:

        x = tf.cast(x, dtype=self.dtype)
        # Encoder
        graph_z_g_x = self.graph_qz_g_x.call(x, training)
        # Decoder
        graph_x_g_z = self.graph_px_g_z.call(graph_z_g_x, training)

        return VaeNet.Output(graph_z_g_x, graph_x_g_z)

    @classmethod
    def call_to_dict(self, result):
        return result._asdict()

    @tf.function
    def call_dict(self, x, training=False):
        return self.call_to_dict(self.call(x, training))

    @property
    def dim_latent(self):
        return self.graph_qz_g_x.latent_dimension

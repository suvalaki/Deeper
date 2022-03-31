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

from deeper.models.vae.base import (
    MultipleObjectiveDimensions,
)

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

        latent_epsilon: float = 1e-6

        enc_mu_embedding_kernel_initializer: Union[
            str, tf.keras.initializers.Initializer
        ] = "he_uniform"
        enc_mu_embedding_bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros"
        enc_mu_latent_kernel_initialiazer: Union[
            str, tf.keras.initializers.Initializer
        ] = "random_normal"
        enc_mu_latent_bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros"

        enc_var_embedding_kernel_initializer: Union[
            str, tf.keras.initializers.Initializer
        ] = "he_uniform"
        enc_var_embedding_bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros"
        enc_var_latent_kernel_initialiazer: Union[
            str, tf.keras.initializers.Initializer
        ] = "random_normal"
        enc_var_latent_bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros"

        recon_embedding_kernel_initializer: Union[
            str, tf.keras.initializers.Initializer
        ] = "he_uniform"
        recon_embedding_bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros"
        recon_latent_kernel_initialiazer: Union[
            str, tf.keras.initializers.Initializer
        ] = "random_normal"
        recon_latent_bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros"

        connected_weights: bool = True
        latent_mu_embedding_dropout: Optional[float] = 0.0
        latent_var_embedding_dropout: Optional[float] = 0.0
        recon_dropout: Optional[float] = 0.0
        latent_fixed_var: Optional[float] = None

        class Config:
            arbitrary_types_allowed = True

    class Output(tf.experimental.ExtensionType, ExtensionTypeIterableMixin):
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
                embedding_dimensions=config.encoder_embedding_dimensions,
                embedding_activations=config.embedding_activations,
                bn_before=config.bn_before,
                bn_after=config.bn_after,
                epsilon=config.latent_epsilon,
                embedding_mu_kernel_initializer=config.enc_mu_embedding_kernel_initializer,
                embedding_mu_bias_initializer=config.enc_mu_embedding_bias_initializer,
                latent_mu_kernel_initialiazer=config.enc_mu_latent_kernel_initialiazer,
                latent_mu_bias_initializer=config.enc_mu_latent_bias_initializer,
                embedding_var_kernel_initializer=config.enc_var_embedding_kernel_initializer,
                embedding_var_bias_initializer=config.enc_var_embedding_bias_initializer,
                latent_var_kernel_initialiazer=config.enc_var_latent_kernel_initialiazer,
                latent_var_bias_initializer=config.enc_var_latent_bias_initializer,
                connected_weights=config.connected_weights,
                embedding_mu_dropout=config.latent_mu_embedding_dropout,
                embedding_var_dropout=config.latent_var_embedding_dropout,
                fixed_var=config.latent_fixed_var,
            ),
            **kwargs,
        )

        # Decoder
        self.graph_px_g_z = VaeReconstructionNet(
            VaeReconstructionNet.Config(
                output_dimensions=config.output_dimensions,
                decoder_embedding_dimensions=config.decoder_embedding_dimensions,
                embedding_activations=config.embedding_activations,
                bn_before=config.bn_before,
                bn_after=config.bn_after,
                recon_embedding_kernel_initializer=config.recon_embedding_kernel_initializer,
                recon_embedding_bias_initializer=config.recon_embedding_bias_initializer,
                recon_latent_kernel_initialiazer=config.recon_latent_kernel_initialiazer,
                recon_latent_bias_initializer=config.recon_latent_bias_initializer,
                recon_dropouut=config.recon_dropout,
            ),
            **kwargs,
        )

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

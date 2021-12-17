from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Activation

from typing import Tuple, Union, Optional, Sequence
from dataclasses import dataclass

from deeper.layers.encoder import Encoder
from deeper.probability_layers.normal import RandomNormalEncoder
from deeper.probability_layers.ops.normal import std_normal_kl_divergence
from deeper.layers.data_splitter import split_inputs, unpack_dimensions
from deeper.utils.function_helpers.decorators import inits_args
from deeper.models.vae.encoder import VaeEncoderNet
from deeper.models.vae.decoder import VaeReconstructionNet

from deeper.probability_layers.normal import (
    lognormal_kl,
    lognormal_pdf,
)
from tensorflow.python.keras.metrics import categorical_accuracy, accuracy

from deeper.utils.tf.keras.models import Model
from typing import Sequence

from typing import NamedTuple
import numpy as np


def reduce_groups(fn, x_grouped: Sequence[tf.Tensor]):
    if len(x_grouped) <= 1:
        return fn(x_grouped[0])
    return tf.concat([fn(z) for z in x_grouped], -1)


class VaeNet(Layer):
    @dataclass
    class Config:
        regression_dimension: int
        boolean_dimension: int
        ordinal_dimension: Union[int, Sequence[int]]
        categorical_dimension: Union[int, Sequence[int]]
        encoder_config: VaeEncoderNet.Config
        decoder_config: VaeReconstructionNet.Config

    class Output(NamedTuple):
        # Encoder/Latent variables
        qz_g_x: VaeEncoderNet.Output
        # DecoderVariables
        px_g_z: VaeReconstructionNet.Output

    def __init__(
        self,
        input_regression_dimension: int,
        input_boolean_dimension: int,
        input_ordinal_dimension: Union[int, Sequence[int]],
        input_categorical_dimension: Union[int, Sequence[int]],
        output_regression_dimension: int,
        output_boolean_dimension: int,
        output_ordinal_dimension: Union[int, Sequence[int]],
        output_categorical_dimension: Union[int, Sequence[int]],
        encoder_embedding_dimensions: Tuple[int],
        decoder_embedding_dimensions: Tuple[int],
        latent_dim: int,
        embedding_activations=tf.nn.relu,
        bn_before: bool = False,
        bn_after: bool = False,
        latent_epsilon=0.0,
        enc_mu_embedding_kernel_initializer="he_uniform",
        enc_mu_embedding_bias_initializer="zeros",
        enc_mu_latent_kernel_initialiazer="random_normal",
        enc_mu_latent_bias_initializer="zeros",
        enc_var_embedding_kernel_initializer="he_uniform",
        enc_var_embedding_bias_initializer="zeros",
        enc_var_latent_kernel_initialiazer="random_normal",
        enc_var_latent_bias_initializer="zeros",
        recon_embedding_kernel_initializer="he_uniform",
        recon_embedding_bias_initializer="zeros",
        recon_latent_kernel_initialiazer="random_normal",
        recon_latent_bias_initializer="zeros",
        connected_weights: bool = True,
        latent_mu_embedding_dropout: Optional[float] = None,
        latent_var_embedding_dropout: Optional[float] = None,
        recon_dropouut: Optional[float] = None,
        latent_fixed_var: Optional[float] = None,
        **kwargs,
    ):
        Layer.__init__(self, **kwargs)

        # Input Dimension Calculation
        (
            self.input_ordinal_dimension,
            self.input_ordinal_dimension_tot,
            self.input_categorical_dimension,
            self.input_categorical_dimension_tot,
            self.input_dim,
        ) = unpack_dimensions(
            input_regression_dimension,
            input_boolean_dimension,
            input_ordinal_dimension,
            input_categorical_dimension,
        )

        # Encoder
        self.graph_qz_g_x = VaeEncoderNet(
            latent_dimension=latent_dim,
            embedding_dimensions=encoder_embedding_dimensions,
            embedding_activations=embedding_activations,
            bn_before=bn_before,
            bn_after=bn_after,
            epsilon=latent_epsilon,
            embedding_mu_kernel_initializer=enc_mu_embedding_kernel_initializer,
            embedding_mu_bias_initializer=enc_mu_embedding_bias_initializer,
            latent_mu_kernel_initialiazer=enc_mu_latent_kernel_initialiazer,
            latent_mu_bias_initializer=enc_mu_latent_bias_initializer,
            embedding_var_kernel_initializer=enc_var_embedding_kernel_initializer,
            embedding_var_bias_initializer=enc_var_embedding_bias_initializer,
            latent_var_kernel_initialiazer=enc_var_latent_kernel_initialiazer,
            latent_var_bias_initializer=enc_var_latent_bias_initializer,
            connected_weights=connected_weights,
            embedding_mu_dropout=latent_mu_embedding_dropout,
            embedding_var_dropout=latent_var_embedding_dropout,
            fixed_var=latent_fixed_var,
            **kwargs,
        )

        # Decoder
        self.graph_px_g_z = VaeReconstructionNet(
            output_regression_dimension=output_regression_dimension,
            output_boolean_dimension=output_boolean_dimension,
            output_ordinal_dimension=output_ordinal_dimension,
            output_categorical_dimension=output_categorical_dimension,
            decoder_embedding_dimensions=decoder_embedding_dimensions,
            embedding_activations=embedding_activations,
            bn_before=bn_before,
            bn_after=bn_after,
            recon_embedding_kernel_initializer=recon_embedding_kernel_initializer,
            recon_embedding_bias_initializer=recon_embedding_bias_initializer,
            recon_latent_kernel_initialiazer=recon_latent_kernel_initialiazer,
            recon_latent_bias_initializer=recon_latent_bias_initializer,
            recon_dropouut=recon_dropouut,
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

    def split_inputs(self, x) -> SplitCovariates:
        return split_inputs(
            x,
            self.input_regression_dimension,
            self.input_boolean_dimension,
            self.input_ordinal_dimension,
            self.input_categorical_dimension,
        )

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
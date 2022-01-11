from __future__ import annotations
import tensorflow as tf

from typing import Optional, Union, Sequence, Tuple

import tensorflow.keras as tfk

Model = tfk.Model

from tensorflow.keras.layers import Layer

from deeper.models.vae import (
    VaeEncoderNet,
    VaeLossNetLatent,
    VaeReconstructionNet,
    VaeReconLossNet,
    VaeNet,
    VaeLossNet,
)

from deeper.probability_layers.ops.normal import (
    lognormal_pdf,
    normal_kl,
)

from typing import NamedTuple
from pydantic.dataclasses import dataclass
from pydantic import Field
from pydantic import BaseModel


class MarginalGmVaeNet(VaeNet):

    """
    The gaussian mixture model is defined by the generative process

        ZgY ~ N(mu_y, szgy)
        X   ~ N(zgy, sx)

    """

    class Config(BaseModel):
        input_regression_dimension: int = Field()
        input_boolean_dimension: int = Field()
        input_ordinal_dimension: Union[int, Sequence[int]] = Field()
        input_categorical_dimension: Union[int, Sequence[int]] = Field()
        output_regression_dimension: int = Field()
        output_boolean_dimension: int = Field()
        output_ordinal_dimension: Union[int, Sequence[int]] = Field()
        output_categorical_dimension: Union[int, Sequence[int]] = Field()
        encoder_embedding_dimensions: Sequence[int] = Field()
        decoder_embedding_dimensions: Sequence[int] = Field()
        latent_dim: int = Field()
        embedding_activations = tf.keras.layers.ReLU()
        bn_before: bool = False
        bn_after: bool = False
        latent_epsilon = 0.0
        enc_mu_embedding_kernel_initializer = "glorot_uniform"
        enc_mu_embedding_bias_initializer = "zeros"
        enc_mu_latent_kernel_initialiazer = "glorot_uniform"
        enc_mu_latent_bias_initializer = "zeros"
        enc_var_embedding_kernel_initializer = "glorot_uniform"
        enc_var_embedding_bias_initializer = "zeros"
        enc_var_latent_kernel_initialiazer = "glorot_uniform"
        enc_var_latent_bias_initializer = "zeros"
        posterior_mu_embedding_kernel_initializer = "glorot_uniform"
        posterior_mu_embedding_bias_initializer = "zeros"
        posterior_mu_latent_kernel_initialiazer = "glorot_uniform"
        posterior_mu_latent_bias_initializer = "zeros"
        posterior_var_embedding_kernel_initializer = "glorot_uniform"
        posterior_var_embedding_bias_initializer = "zeros"
        posterior_var_latent_kernel_initialiazer = "glorot_uniform"
        posterior_var_latent_bias_initializer = "zeros"
        recon_embedding_kernel_initializer = "glorot_uniform"
        recon_embedding_bias_initializer = "zeros"
        recon_latent_kernel_initialiazer = "glorot_uniform"
        recon_latent_bias_initializer = "zeros"
        connected_weights: bool = True
        latent_mu_embedding_dropout: Optional[float] = None
        latent_var_embedding_dropout: Optional[float] = None
        posterior_mu_dropout: Optional[float] = None
        posterior_var_dropout: Optional[float] = None
        recon_dropouut: Optional[float] = None
        latent_fixed_var: Optional[float] = None

        class Config:
            arbitrary_types_allowed = True

    class Output(NamedTuple):
        # Encoder/Latent variables
        qz_g_xy: VaeEncoderNet.Output
        # DecoderVariables
        px_g_zy: VaeReconstructionNet.Output
        pz_g_y: VaeEncoderNet.Output

    @classmethod
    def from_config(cls, config: MarginalGmVaeNet.Config, **kwargs):
        return cls(**config, **kwargs)

    def __init__(
        self,
        config: MarginalGmVaeNet.Config,
        **kwargs,
    ):

        super().__init__(
            input_regression_dimension=config.input_regression_dimension,
            input_boolean_dimension=config.input_boolean_dimension,
            input_ordinal_dimension=config.input_ordinal_dimension,
            input_categorical_dimension=config.input_categorical_dimension,
            output_regression_dimension=config.output_regression_dimension,
            output_boolean_dimension=config.output_boolean_dimension,
            output_ordinal_dimension=config.output_ordinal_dimension,
            output_categorical_dimension=config.output_categorical_dimension,
            encoder_embedding_dimensions=config.encoder_embedding_dimensions,
            decoder_embedding_dimensions=config.decoder_embedding_dimensions,
            latent_dim=config.latent_dim,
            embedding_activations=config.embedding_activations,
            enc_mu_embedding_kernel_initializer=config.enc_mu_embedding_kernel_initializer,
            enc_mu_embedding_bias_initializer=config.enc_mu_embedding_bias_initializer,
            enc_mu_latent_kernel_initialiazer=config.enc_mu_latent_kernel_initialiazer,
            enc_mu_latent_bias_initializer=config.enc_mu_latent_bias_initializer,
            enc_var_embedding_kernel_initializer=config.enc_var_embedding_kernel_initializer,
            enc_var_embedding_bias_initializer=config.enc_var_embedding_bias_initializer,
            enc_var_latent_kernel_initialiazer=config.enc_var_latent_kernel_initialiazer,
            enc_var_latent_bias_initializer=config.enc_var_latent_bias_initializer,
            recon_embedding_kernel_initializer=config.recon_embedding_kernel_initializer,
            recon_embedding_bias_initializer=config.recon_embedding_bias_initializer,
            recon_latent_kernel_initialiazer=config.recon_latent_kernel_initialiazer,
            recon_latent_bias_initializer=config.recon_latent_bias_initializer,
            connected_weights=config.connected_weights,
            latent_mu_embedding_dropout=config.latent_mu_embedding_dropout,
            latent_var_embedding_dropout=config.latent_var_embedding_dropout,
            recon_dropouut=config.recon_dropouut,
            latent_fixed_var=config.latent_fixed_var,
            **kwargs,
        )
        self.graph_pz_g_y = VaeEncoderNet(
            latent_dimension=config.latent_dim,
            embedding_dimensions=[],
            bn_before=config.bn_before,
            bn_after=config.bn_after,
            epsilon=0.0,
            embedding_mu_kernel_initializer=config.posterior_mu_embedding_kernel_initializer,
            embedding_mu_bias_initializer=config.posterior_mu_embedding_bias_initializer,
            latent_mu_kernel_initialiazer=config.posterior_mu_latent_kernel_initialiazer,
            latent_mu_bias_initializer=config.posterior_mu_latent_bias_initializer,
            embedding_var_kernel_initializer=config.posterior_var_embedding_kernel_initializer,
            embedding_var_bias_initializer=config.posterior_var_embedding_bias_initializer,
            latent_var_kernel_initialiazer=config.posterior_var_latent_kernel_initialiazer,
            latent_var_bias_initializer=config.posterior_var_latent_bias_initializer,
            connected_weights=config.connected_weights,
            embedding_mu_dropout=config.posterior_mu_dropout,
            embedding_var_dropout=config.posterior_var_dropout,
            fixed_var=config.latent_fixed_var,
        )

    # def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training=False):
    # @tf.function
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training=False):
        # v is the input data. y is the category.
        v, y = inputs
        x = vy = tf.concat([v, y], axis=-1)
        qz_g_xy, px_g_zy = super().call(vy, training)
        pz_g_y = self.graph_pz_g_y.call(y, training)
        return self.Output(qz_g_xy, px_g_zy, pz_g_y)

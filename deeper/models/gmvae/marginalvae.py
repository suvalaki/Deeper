from __future__ import annotations
import tensorflow as tf

from typing import Optional, Union, Sequence, Tuple

import tensorflow.keras as tfk

Model = tfk.Model

from tensorflow.keras.layers import Layer

from deeper.models.vae import (
    MultipleObjectiveDimensions,
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
from deeper.utils.tf.experimental.extension_type import ExtensionTypeIterableMixin


class MarginalGmVaeNet(VaeNet):

    """
    The gaussian mixture model is defined by the generative process

        ZgY ~ N(mu_y, szgy)
        X   ~ N(zgy, sx)

    """

    class Config(VaeNet.Config):
        posterior_mu_embedding_kernel_initializer = "glorot_uniform"
        posterior_mu_embedding_bias_initializer = "zeros"
        posterior_mu_latent_kernel_initialiazer = "glorot_uniform"
        posterior_mu_latent_bias_initializer = "zeros"
        posterior_var_embedding_kernel_initializer = "glorot_uniform"
        posterior_var_embedding_bias_initializer = "zeros"
        posterior_var_latent_kernel_initialiazer = "glorot_uniform"
        posterior_var_latent_bias_initializer = "zeros"
        posterior_mu_dropout: Optional[float] = 0.0
        posterior_var_dropout: Optional[float] = 0.0

    class Output(tf.experimental.ExtensionType, ExtensionTypeIterableMixin):
        # Encoder/Latent variables
        qz_g_xy: VaeEncoderNet.Output
        # DecoderVariables
        px_g_zy: VaeReconstructionNet.Output
        pz_g_y: VaeEncoderNet.Output

        # TODO: make the encoder a concat of qzgxy and pzgy instead of just
        # allowing the encoder to flow?

        @property 
        def encoder(self):
            return self.pz_g_y

        @property 
        def decoder(self):
            return self.px_g_zy

    @classmethod
    def from_config(cls, config: MarginalGmVaeNet.Config, **kwargs):
        return cls(**config, **kwargs)

    def __init__(
        self,
        config: MarginalGmVaeNet.Config,
        **kwargs,
    ):

        super().__init__(config, **kwargs)
        self.graph_pz_g_y = VaeEncoderNet(
            VaeEncoderNet.Config(
                latent_dim=self.config.latent_dim,
                embedding_dimensions=[],
                bn_before=self.config.bn_before,
                bn_after=self.config.bn_after,
                epsilon=0.0,
                embedding_mu_kernel_initializer=config.posterior_mu_embedding_kernel_initializer,
                embedding_mu_bias_initializer=config.posterior_mu_embedding_bias_initializer,
                latent_mu_kernel_initialiazer=config.posterior_mu_latent_kernel_initialiazer,
                latent_mu_bias_initializer=config.posterior_mu_latent_bias_initializer,
                embedding_var_kernel_initializer=config.posterior_var_embedding_kernel_initializer,
                embedding_var_bias_initializer=config.posterior_var_embedding_bias_initializer,
                latent_var_kernel_initialiazer=config.posterior_var_latent_kernel_initialiazer,
                latent_var_bias_initializer=config.posterior_var_latent_bias_initializer,
                connected_weights=self.graph_qz_g_x.connected_weights,
                embedding_mu_dropout=config.posterior_mu_dropout,
                embedding_var_dropout=config.posterior_var_dropout,
                fixed_var=self.graph_qz_g_x.fixed_var,
            ),
            **kwargs,
        )

    # def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training=False):
    @tf.function
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training=False):
        # v is the input data. y is the category.
        v, y = inputs
        x = vy = tf.concat([v, y], axis=-1)
        qz_g_xy, px_g_zy = super().call(vy, training)
        pz_g_y = self.graph_pz_g_y.call(y, training)
        return self.Output(qz_g_xy, px_g_zy, pz_g_y)

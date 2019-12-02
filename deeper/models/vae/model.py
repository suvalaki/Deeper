import tensorflow as tf
import numpy as np

from deeper.utils.scope import Scope
from deeper.layers.random_normal import NormalEncoder
from deeper.layers.binary import SigmoidEncoder
from deeper.ops.distance import std_normal_kl_divergence

tfk = tf.keras
Model = tfk.Model
Layer = tfk.layers.Layer


class VAE(Model):
    @inits_args
    def __init__(
        self,
        input_dimension,
        embedding_dimensions,
        latent_dim,
        embedding_activations=tf.nn.tanh,
        kind="binary",
        var_scope="marginal_autoencoder",
        bn_before=False,
        bn_after=False,
        latent_epsilon=0.0,
        reconstruction_epsilon=0.0,
        enc_mu_embedding_kernel_initializer="glorot_uniform",
        enc_mu_embedding_bias_initializer="zeros",
        enc_mu_latent_kernel_initialiazer="glorot_uniform",
        enc_mu_latent_bias_initializer="zeros",
        enc_var_embedding_kernel_initializer="glorot_uniform",
        enc_var_embedding_bias_initializer="zeros",
        enc_var_latent_kernel_initialiazer="glorot_uniform",
        enc_var_latent_bias_initializer="zeros",
        recon_embedding_kernel_initializer="glorot_uniform",
        recon_embedding_bias_initializer="zeros",
        recon_latent_kernel_initialiazer="glorot_uniform",
        recon_latent_bias_initializer="zeros",
        connected_weights=True,
        latent_mu_embedding_dropout=0.0,
        latent_var_embedding_dropout=0.0,
        recon_dropouut=0.0,
        latent_fixed_var=None,
    ):
        Model.__init__(self)
        Scope.__init__(self, var_scope)

        # Encoder
        self.graph_qz_g_x = RandomNormalEncoder(
            latent_dimension=self.latent_dim,
            embedding_dimensions=self.embedding_dimensions,
            var_scope=self.v_name("graph_qz_g_x"),
            bn_before=self.bn_before,
            bn_after=self.bn_after,
            epsilon=self.latent_epsilon,
            embedding_mu_kernel_initializer=latent_mu_embedding_kernel_initializer,
            embedding_mu_bias_initializer=latent_mu_embedding_bias_initializer,
            latent_mu_kernel_initialiazer=latent_mu_latent_kernel_initialiazer,
            latent_mu_bias_initializer=latent_mu_latent_bias_initializer,
            embedding_var_kernel_initializer=latent_var_embedding_kernel_initializer,
            embedding_var_bias_initializer=latent_var_embedding_bias_initializer,
            latent_var_kernel_initialiazer=latent_var_latent_kernel_initialiazer,
            latent_var_bias_initializer=latent_var_latent_bias_initializer,
            connected_weights=connected_weights,
            embedding_mu_dropout=latent_mu_embedding_dropout,
            embedding_var_dropout=latent_var_embedding_dropout,
            fixed_var=latent_fixed_var,
        )

        # Decoder
        if self.kind == "binary":
            self.graph_px_g_z = SigmoidEncoder(
                latent_dimension=self.input_dimension,
                embedding_dimensions=self.embedding_dimensions[::-1],
                var_scope=self.v_name("graph_px_g_z"),
                bn_before=self.bn_before,
                bn_after=self.bn_after,
                epsilon=self.reconstruction_epsilon,
                embedding_kernel_initializer=recon_embedding_kernel_initializer,
                embedding_bias_initializer=recon_embedding_bias_initializer,
                latent_kernel_initialiazer=recon_latent_kernel_initialiazer,
                latent_bias_initializer=recon_latent_bias_initializer,
                embedding_dropout=recon_dropouut,
            )
        else:
            self.graph_px_g_z = RandomNormalEncoder(
                self.input_dimension,
                self.embedding_dimensions[::-1],
                var_scope=self.v_name("graph_px_g_z"),
                bn_before=self.bn_before,
                bn_after=self.bn_after,
                embedding_mu_dropout=recon_dropouut,
                embedding_var_dropout=recon_dropouut,
                fixed_var=1.0,
                epsilon=self.reconstruction_epsilon,
                embedding_mu_kernel_initializer=recon_embedding_kernel_initializer,
                embedding_mu_bias_initializer=recon_embedding_bias_initializer,
                latent_mu_kernel_initialiazer=recon_latent_kernel_initialiazer,
                latent_mu_bias_initializer=recon_latent_bias_initializer,
            )

    # @tf.function#
    def call(self, x, training=False):
        x = tf.cast(x, dtype=self.dtype)
        (
            qz_g_xy__sample,
            qz_g_xy__logprob,
            qz_g_xy__prob,
            qz_g_xy__mu,
            qz_g_xy__logvar,
            qz_g_xy__var,
        ) = self.graph_qz_g_x.call(x, training)

        (
            px_g_zy__sample,
            px_g_zy__logprob,
            px_g_zy__prob,
        ) = self.graph_px_g_z.call(qz_g_x__sample, training, x)[0:3]

        recon = px_g_zy__logprob
        z_entropy = std_normal_kl_divergence(qz_g_xy__mu, qz_g_xy__logvar)
        elbo = recon + z_entropy

        output = {
            "qz_g_xy__sample": qz_g_xy__sample,
            "qz_g_xy__logprob": qz_g_xy__logprob,
            "qz_g_xy__prob": qz_g_xy__prob,
            "qz_g_xy__mu": qz_g_xy__mu,
            "qz_g_xy__logvar": qz_g_xy__logvar,
            "qz_g_xy__var": qz_g_xy__var,
            "px_g_zy__sample": px_g_zy__sample,
            "px_g_zy__logprob": px_g_zy__logprob,
            "px_g_zy__prob": px_g_zy__prob,
            "recon": recon,
            "z_entropy": z_entropy,
            "elbo": elbo,
        }

        return output

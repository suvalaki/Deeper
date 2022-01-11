from __future__ import annotations
import tensorflow as tf

from typing import Union, Tuple, Sequence, Optional, NamedTuple
from pydantic.dataclasses import dataclass

from deeper.models.gmvae.gmvae_pure_sampling.network import GumbleGmvaeNet
from deeper.models.gmvae.marginalvae_loss import MarginalGmVaeLossNet
from deeper.layers.categorical import CategoricalEncoder
from deeper.models.vae.network_loss import VaeLossNet


class GumbleGmvaeNetLossNet(tf.keras.layers.Layer):

    class InputWeight(NamedTuple):
        lambda_y: float  = 1.0
        lambda_z: float = 1.0
        lambda_reg: float = 1.0
        lambda_bin: float = 1.0
        lambda_ord: float = 1.0
        lambda_cat: float = 1.0

    class Input(NamedTuple):
        py: tf.Tensor 
        qy_g_x: CategoricalEncoder.Output
        marginal: MarginalGmVaeLossNet.Input
        weight_component: tf.Tensor

        @staticmethod 
        def from_GumbleGmvaeNet_output(
            y_true: SplitCovariates,
            model_output: GumbleGmvaeNet.Output,
            weights: GumbleGmvaeNetLossNet.InputWeight,
        ) -> MarginalGmVaeLossNet.Input:
            return GumbleGmvaeNetLossNet.Input(
                model_output.py, 
                model_output.qy_g_x, 
                MarginalGmVaeLossNet.Input.from_MarginalGmVae_output(
                    y_true, model_output.marginal, 
                    VaeLossNet.InputWeight(*weights[1:])
                ),
                weights[0]
            )

    class Output(NamedTuple):
        # y losses
        kl_y: tf.Tensor
        # marginal losses
        kl_zgy: tf.Tensor
        l_pxgzy_reg: tf.Tensor
        l_pxgzy_bin: tf.Tensor
        l_pxgzy_ord: tf.Tensor
        l_pxgzy_cat: tf.Tensor
        scaled_elbo: tf.Tensor
        recon_loss: tf.Tensor
        loss: tf.Tensor
        lambda_z: tf.Tensor
        lambda_reg: tf.Tensor
        lambda_bin: tf.Tensor
        lambda_ord: tf.Tensor
        lambda_cat: tf.Tensor

    def __init__(
        self, 
        cat_latent_eps = 0.0, 
        latent_eps = 0.0, 
        posterior_eps = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.marginal_lossnet = MarginalGmVaeLossNet(
            latent_eps, posterior_eps, **kwargs)

    def call(self, inputs: GumbleGmvaeNetLossNet.Input, training:bool=False):
        losses_marginal = self.marginal_lossnet(inputs.marginal)
        
        # y_entropy
        # E_q [log(p/q)] = sum q (log_p - log_q)
        y_entropy = tf.reduce_sum(
            inputs.qy_g_x.probs * tf.math.log(inputs.py), -1
        ) + tf.nn.softmax_cross_entropy_with_logits(
            logits = inputs.qy_g_x.logits, 
            labels=inputs.qy_g_x.probs
        )

        # loss = recon + z_ent + y_ent
        # loss = E[px_g_zy__logprob] + E[pz_g_y__logprob - qz_g_xy__logprob] + E_q [log(p/q)]
        scaled_elbo = losses_marginal.scaled_elbo + inputs.weight_component* y_entropy
        loss = -scaled_elbo

        return GumbleGmvaeNetLossNet.Output(
            kl_y = y_entropy,
            kl_zgy = losses_marginal.kl_zgy,
            l_pxgzy_reg = losses_marginal.l_pxgzy_reg,
            l_pxgzy_bin = losses_marginal.l_pxgzy_bin, 
            l_pxgzy_ord = losses_marginal.l_pxgzy_ord,
            l_pxgzy_cat = losses_marginal.l_pxgzy_cat,
            scaled_elbo = scaled_elbo,
            recon_loss = losses_marginal.recon_loss,
            loss = loss,
            lambda_z = losses_marginal.lambda_z,
            lambda_reg = losses_marginal.lambda_reg,
            lambda_bin = losses_marginal.lambda_bin,
            lambda_ord = losses_marginal.lambda_ord,
            lambda_cat = losses_marginal.lambda_cat,
        )
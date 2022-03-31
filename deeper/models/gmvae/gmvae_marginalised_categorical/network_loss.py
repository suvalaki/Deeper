from __future__ import annotations
import tensorflow as tf

from typing import Union, Tuple, Sequence, Optional, NamedTuple
from pydantic.dataclasses import dataclass

from deeper.models.gmvae.base import GmvaeNetLossNetBase
from deeper.models.gmvae.gmvae_marginalised_categorical.network import (
    StackedGmvaeNet,
)
from deeper.models.gmvae.marginalvae_loss import MarginalGmVaeLossNet
from deeper.layers.categorical import CategoricalEncoder
from deeper.models.vae.network_loss import VaeLossNet
from deeper.utils.tf.experimental.extension_type import ExtensionTypeIterableMixin


class StackedGmvaeLossNet(GmvaeNetLossNetBase):
    class Input(tf.experimental.ExtensionType, ExtensionTypeIterableMixin):
        py: tf.Tensor
        qy_g_x: CategoricalEncoder.Output
        marginals: Tuple[MarginalGmVaeLossNet.Input, ...]
        weight_component: tf.Tensor

        @staticmethod
        def from_output(
            y_true: SplitCovariates,
            model_output: StackedGmvaeNet.Output,
            weights: StackedGmvaeLossNet.InputWeight,
        ) -> MarginalGmVaeLossNet.Input:
            return StackedGmvaeLossNet.Input(
                model_output.py,
                model_output.qy_g_x,
                [
                    MarginalGmVaeLossNet.Input.from_MarginalGmVae_output(
                        y_true, marg, VaeLossNet.InputWeight(*weights[1:])
                    )
                    for marg in model_output.marginals
                ],
                weights.lambda_y,
            )

    def __init__(
        self,
        cat_latent_eps=0.0,
        latent_eps=0.0,
        posterior_eps=0.0,
        encoder_name="zgy",
        decoder_name="xgz",
        prefix="",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.marginal_lossnet = MarginalGmVaeLossNet(
            latent_eps, posterior_eps, encoder_name, decoder_name, prefix, **kwargs
        )

    @tf.function
    def call(self, inputs: StackedGmvaeLossNet.Input, training: bool = False):

        # Component losses
        marginal_losses = [self.marginal_lossnet(marginal) for marginal in inputs.marginals]
        combined_losses = MarginalGmVaeLossNet.Output(
            *[
                tf.reduce_sum(inputs.qy_g_x.probs * tf.stack(x, axis=-1), axis=-1)
                for x in list(map(list, zip(*marginal_losses)))
            ]
        )

        # y_entropy
        # E_q [log(p/q)] = sum q (log_p - log_q)
        # y_entropy = tf.reduce_sum(
        #     inputs.qy_g_x.probs * tf.math.log(inputs.py), -1
        # ) + tf.nn.softmax_cross_entropy_with_logits(
        #     logits = inputs.qy_g_x.logits,
        #     labels = inputs.qy_g_x.probs
        # )

        y_entropy = tf.reduce_sum(
            inputs.qy_g_x.probs
            * (tf.math.log(inputs.py) - tf.nn.log_softmax(logits=inputs.qy_g_x.logits)),
            -1,
        )

        self.add_metric(tf.reduce_sum(y_entropy), name="y_entropy")

        # loss = recon + z_ent + y_ent
        # loss = E[px_g_zy__logprob] + E[pz_g_y__logprob - qz_g_xy__logprob] + E_q [log(p/q)]
        scaled_elbo = combined_losses.scaled_elbo + inputs.weight_component * y_entropy
        loss = -scaled_elbo

        return StackedGmvaeLossNet.Output(
            kl_y=y_entropy,
            kl_zgy=combined_losses.kl_zgy,
            l_pxgzy_reg=combined_losses.l_pxgzy_reg,
            l_pxgzy_bin=combined_losses.l_pxgzy_bin,
            l_pxgzy_ord=combined_losses.l_pxgzy_ord,
            l_pxgzy_cat=combined_losses.l_pxgzy_cat,
            scaled_l_pxgzy=combined_losses.scaled_l_pxgzy,
            scaled_elbo=scaled_elbo,
            recon_loss=combined_losses.recon_loss,
            loss=loss,
            lambda_z=combined_losses.lambda_z,
            lambda_reg=combined_losses.lambda_reg,
            lambda_bin=combined_losses.lambda_bin,
            lambda_ord=combined_losses.lambda_ord,
            lambda_cat=combined_losses.lambda_cat,
        )

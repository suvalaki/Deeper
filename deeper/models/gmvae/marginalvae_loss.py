from __future__ import annotations
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Layer

from typing import Optional, Union, Sequence, Tuple, NamedTuple


from deeper.models.vae import VaeLossNet
from deeper.models.vae.encoder_loss import VaeLossNetLatent
from deeper.models.vae.decoder_loss import VaeReconLossNet
from deeper.models.vae.utils import SplitCovariates
from deeper.probability_layers.ops.normal import normal_kl, lognormal_pdf, normal_kl
from deeper.utils.tf.experimental.extension_type import ExtensionTypeIterableMixin


class MarginalGmVaeLossNet(VaeLossNet):
    def __init__(
        self,
        latent_eps=0.0,
        posterior_eps=0.0,
        encoder_name="qzgy",
        decoder_name="pxgz",
        prefix="",
        **kwargs
    ):
        super(MarginalGmVaeLossNet, self).__init__(
            latent_eps=latent_eps, prefix="marginal_ae", **kwargs
        )
        self.posterior_eps = posterior_eps

    class Input(tf.experimental.ExtensionType, ExtensionTypeIterableMixin):
        latent_sample: tf.Tensor
        prior_latent: VaeLossNetLatent.Input
        posterior_latent: VaeLossNetLatent.Input
        y_true: VaeReconLossNet.InputYTrue
        y_pred: VaeReconLossNet.InputYPred
        weight: VaeLossNet.InputWeight

        @staticmethod
        def from_MarginalGmVae_output(
            y_true: SplitCovariates,
            model_output: MarginalGmVaeNet.Output,
            weights: VaeLossNet.InputWeight,
        ) -> MarginalGmVaeLossNet.Input:
            return MarginalGmVaeLossNet.Input(
                model_output.qz_g_xy.sample,
                VaeLossNetLatent.Input(model_output.pz_g_y.mu, model_output.pz_g_y.logvar),
                VaeLossNetLatent.Input(model_output.qz_g_xy.mu, model_output.qz_g_xy.logvar),
                VaeReconLossNet.InputYTrue(
                    y_true.regression,
                    y_true.binary,
                    y_true.ordinal_groups,
                    y_true.categorical_groups,
                ),
                VaeReconLossNet.InputYPred.from_VaeReconstructionNet(model_output.px_g_zy),
                weights,
            )

    class Output(tf.experimental.ExtensionType, ExtensionTypeIterableMixin):
        # renamed outputs
        kl_zgy: tf.Tensor
        l_pxgzy_reg: tf.Tensor
        l_pxgzy_bin: tf.Tensor
        l_pxgzy_ord: tf.Tensor
        l_pxgzy_cat: tf.Tensor
        scaled_l_pxgzy: tf.Tensor
        scaled_elbo: tf.Tensor
        recon_loss: tf.Tensor
        loss: tf.Tensor
        lambda_z: tf.Tensor
        lambda_reg: tf.Tensor
        lambda_bin: tf.Tensor
        lambda_ord: tf.Tensor
        lambda_cat: tf.Tensor

    @tf.function
    def latent_kl(self, z, mu_pz, mu_qzgy, logvar_pz, logvar_qzgy, training: bool = False):
        kl = normal_kl(
            z,
            mu_pz,
            mu_qzgy,
            logvar_pz,
            logvar_qzgy,
            self.posterior_eps,
            self.latent_eps,
        )
        self.add_metric(kl, name="{self.prefix}/weight/pzgy_latent_kl")
        return kl

    @tf.function
    def call(
        self, inputs: MarginalGmVaeLossNet.Input, training=False
    ) -> MarginalGmVaeLossNet.Output:
        outs = super().call(VaeLossNet.Input(*inputs[2:]), training)
        # kl_zgy = tf.reduce_sum(
        #     (
        #         lognormal_pdf(
        #             inputs.latent_sample, inputs.prior_latent.mu, inputs.prior_latent.logvar
        #         )
        #         - lognormal_pdf(
        #             inputs.latent_sample, inputs.posterior_latent.mu, inputs.posterior_latent.logvar
        #         )
        #     ),
        #     axis=-1,
        # )

        kl_zgy = -normal_kl(
            inputs.latent_sample,
            inputs.posterior_latent.mu,
            inputs.prior_latent.mu,
            inputs.posterior_latent.logvar,
            inputs.prior_latent.logvar,
        )

        logprob_recon = outs.l_pxgz_reg + outs.l_pxgz_bin + outs.l_pxgz_ord + outs.l_pxgz_cat
        scaled_elbo = logprob_recon + kl_zgy * inputs.weight.lambda_z
        recon_loss = -logprob_recon
        loss = -scaled_elbo

        losses = MarginalGmVaeLossNet.Output(
            kl_zgy=kl_zgy,
            l_pxgzy_reg=outs.l_pxgz_reg,
            l_pxgzy_bin=outs.l_pxgz_bin,
            l_pxgzy_ord=outs.l_pxgz_ord,
            l_pxgzy_cat=outs.l_pxgz_cat,
            scaled_l_pxgzy=outs.scaled_l_pxgz,
            scaled_elbo=-loss,
            recon_loss=recon_loss,
            loss=loss,
            lambda_z=inputs.weight.lambda_z,
            lambda_reg=inputs.weight.lambda_reg,
            lambda_bin=inputs.weight.lambda_bin,
            lambda_ord=inputs.weight.lambda_ord,
            lambda_cat=inputs.weight.lambda_cat,
        )

        return losses

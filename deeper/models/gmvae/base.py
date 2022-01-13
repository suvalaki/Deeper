from __future__ import annotations
import tensorflow as tf
import tensorflow_addons as tfa
from typing import NamedTuple, Sequence
from pydantic import BaseModel
from deeper.models.vae import Vae
from deeper.models.gmvae.marginalvae import MarginalGmVaeNet
from deeper.models.generalised_autoencoder.base import AutoencoderBase


class GmvaeNetBase(AutoencoderBase):
    class Config(MarginalGmVaeNet.Config):
        components: int = None
        cat_embedding_dimensions: Sequence[int] = None
        cat_embedding_kernel_initializer = "glorot_uniform"
        cat_embedding_bias_initializer = "zeros"
        cat_latent_kernel_initialiazer = "glorot_uniform"
        cat_latent_bias_initializer = "zeros"
        categorical_epsilon = 0.0

    def split_outputs(self, y) -> SplitCovariates:
        return self.graph_marginal_autoencoder.split_outputs(y)


class GmvaeNetLossNetBase(tf.keras.layers.Layer):
    class InputWeight(NamedTuple):
        lambda_y: float = 1.0
        lambda_z: float = 1.0
        lambda_reg: float = 1.0
        lambda_bin: float = 1.0
        lambda_ord: float = 1.0
        lambda_cat: float = 1.0

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
        # weights
        lambda_z: tf.Tensor
        lambda_reg: tf.Tensor
        lambda_bin: tf.Tensor
        lambda_ord: tf.Tensor
        lambda_cat: tf.Tensor


class GmvaeModelBase(tf.keras.Model):
    class WeigtScheduleConfig(Vae.WeigtScheduleConfig):
        kld_y_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = (
            tfa.optimizers.CyclicalLearningRate(
                1.0,
                1.0,
                step_size=1,
                scale_fn=lambda x: 1.0,
                scale_mode="cycle",
            )
        )

    class Config(GmvaeNetBase.Config, WeigtScheduleConfig):
        pass

    _output_keys_renamed = {
        "kl_y": "losses/kl_y",
        "kl_zgy": "losses/kl_zgy",
        "l_pxgzy_reg": "reconstruction/l_pxgzy_reg",
        "l_pxgzy_bin": "reconstruction/l_pxgzy_bin",
        "l_pxgzy_ord": "reconstruction/l_pxgzy_ord",
        "l_pxgzy_cat": "reconstruction/l_pxgzy_cat",
        "scaled_elbo": "losses/scaled_elbo",
        "recon_loss": "losses/recon_loss",
        "loss": "losses/loss",
        "lambda_z": "weight/lambda_z",
        "lambda_reg": "weight/lambda_reg",
        "lambda_bin": "weight/lambda_bin",
        "lambda_ord": "weight/lambda_ord",
        "lambda_cat": "weight/lambda_cat",
        "kld_y_schedule": "weight/lambda_y",
        "kld_z_schedule": "weight/lambda_z",
    }

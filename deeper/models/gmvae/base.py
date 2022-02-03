from __future__ import annotations
import tensorflow as tf
import tensorflow_addons as tfa
from typing import NamedTuple, Sequence
from pydantic import BaseModel
from deeper.models.vae import Vae
from deeper.models.gmvae.marginalvae import MarginalGmVaeNet
from deeper.models.generalised_autoencoder.base import (
    AutoencoderBase,
    AutoencoderModelBaseMixin,
)
from deeper.utils.model_mixins import InputDisentangler, ClusteringMixin

from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop


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
        scaled_l_pxgzy: tf.Tensor
        scaled_elbo: tf.Tensor
        recon_loss: tf.Tensor
        loss: tf.Tensor
        # weights
        lambda_z: tf.Tensor
        lambda_reg: tf.Tensor
        lambda_bin: tf.Tensor
        lambda_ord: tf.Tensor
        lambda_cat: tf.Tensor


class GmvaeModelBase(tf.keras.Model, AutoencoderModelBaseMixin, ClusteringMixin):
    class CoolingRegime(Vae.CoolingRegime):
        class Config(Vae.CoolingRegime.Config):
            kld_y_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = (
                tfa.optimizers.CyclicalLearningRate(
                    1.0,
                    1.0,
                    step_size=1,
                    scale_fn=lambda x: 1.0,
                    scale_mode="cycle",
                )
            )

        def call(self, step):
            cstep = tf.cast(step, self.dtype)
            kld_y_schedule = self.config.kld_y_schedule(cstep)
            kld_z_schedule = self.config.kld_z_schedule(cstep)
            recon_schedule = self.config.recon_schedule(cstep)
            recon_reg_schedule = self.config.recon_reg_schedule(cstep)
            recon_bin_schedule = self.config.recon_bin_schedule(cstep)
            recon_ord_schedule = self.config.recon_ord_schedule(cstep)
            recon_cat_schedule = self.config.recon_cat_schedule(cstep)
            return GmvaeNetLossNetBase.InputWeight(
                kld_y_schedule,
                kld_z_schedule,
                recon_reg_schedule,
                recon_bin_schedule,
                recon_ord_schedule,
                recon_cat_schedule,
            )

    class Config(GmvaeNetBase.Config, CoolingRegime.Config):
        monte_carlo_training_samples: int = 1

    _output_keys_renamed = {
        "kl_y": "losses/kl_y",
        "kl_zgy": "losses/kl_zgy",
        "l_pxgzy_reg": "reconstruction/l_pxgzy_reg",
        "l_pxgzy_bin": "reconstruction/l_pxgzy_bin",
        "l_pxgzy_ord": "reconstruction/l_pxgzy_ord",
        "l_pxgzy_cat": "reconstruction/l_pxgzy_cat",
        "scaled_l_pxgzy": "reconstruction/scaled_l_pxgzy",
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

    def __init__(self, config, **kwargs):
        tf.keras.Model.__init__(self, **kwargs)
        self.config = config
        self.network = config.get_network_type()(config)
        self.lossnet = config.get_lossnet_type()()
        self.weight_getter = config.get_cooling_regime()(config, dtype=self.dtype)
        AutoencoderModelBaseMixin.__init__(
            self,
            self.weight_getter,
            self.network,
            config.get_adversarialae_fake_output_getter()(),
            config.get_fake_output_getter()(),
        )
        ClusteringMixin.__init__(
            self,
            self.weight_getter,
            self.network,
            config.get_cluster_output_parser_type()(),
        )

    def train_step(self, data, training: bool = False):

        data = data_adapter.expand_1d(data)
        x, y = data
        inputs, temp, weights = self.call_inputs(x)

        loss = 0.0
        with backprop.GradientTape() as tape:
            for i in range(self.config.monte_carlo_training_samples):
                y_pred = self.network(inputs, training=True)
                losses = self.loss_fn(
                    y,
                    y_pred,
                    weights,
                    training=True,
                )
                loss += tf.reduce_mean(losses.loss)
            loss /= self.config.monte_carlo_training_samples
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        tempr = {"temp": temp} if "temp" in self._output_keys_renamed else {}
        return {
            self._output_keys_renamed[k]: v
            for k, v in {
                # **{v.name: v.result() for v in self.metrics}
                **losses._asdict(),
                **tempr,
                "kld_y_schedule": weights.lambda_y,
                "kld_z_schedule": weights.lambda_z,
            }.items()
        }

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y = data
        inputs, temp, weights = self.call_inputs(x)

        if temp is not None:
            inpputs = (x, 0.5)

        y_pred = self.network(inputs, training=False)
        losses = self.loss_fn(y, y_pred, self.lossnet.InputWeight(), training=False)
        loss = tf.reduce_mean(losses.loss)

        return {self._output_keys_renamed[k]: v for k, v in losses._asdict().items()}

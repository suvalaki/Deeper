from __future__ import annotations
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from typing import Union, Tuple, Sequence

from deeper.probability_layers.ops.normal import std_normal_kl_divergence
from deeper.layers.encoder import Encoder
from deeper.layers.binary import SigmoidEncoder

# from deeper.probability_layers.gumble_softmax import GumbleSoftmaxLayer
from deeper.probability_layers.normal import (
    RandomNormalEncoder,
    lognormal_kl,
    lognormal_pdf,
)
from deeper.utils.function_helpers.decorators import inits_args
from deeper.utils.function_helpers.collectors import get_local_tensors
from deeper.utils.scope import Scope
from deeper.models.vae.metrics import vae_categorical_dims_accuracy

from deeper.layers.data_splitter import split_inputs
from deeper.models.vae.network import VaeNet
from deeper.models.vae.network_loss import VaeLossNet
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop

from types import SimpleNamespace
from deeper.utils.tf.keras.models import GenerativeModel


from tensorflow.python.keras.metrics import (
    categorical_accuracy,
)

tfk = tf.keras
Layer = tfk.layers.Layer

from collections import namedtuple
from deeper.models.vae.utils import SplitCovariates
from deeper.models.vae.network import VaeNet
from deeper.models.generalised_autoencoder.base import (
    AutoencoderModelBaseMixin,
)

from pydantic import BaseModel


class Vae(tf.keras.Model, AutoencoderModelBaseMixin):
    class CoolingRegime(tf.keras.layers.Layer):
        class Config(BaseModel):
            kld_z_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = (
                tfa.optimizers.CyclicalLearningRate(
                    1.0,
                    1.0,
                    step_size=1,
                    scale_fn=lambda x: 1.0,
                    scale_mode="cycle",
                )
            )
            recon_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = (
                tfa.optimizers.CyclicalLearningRate(
                    1.0,
                    1.0,
                    step_size=1,
                    scale_fn=lambda x: 1.0,
                    scale_mode="cycle",
                )
            )
            recon_reg_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = (
                tfa.optimizers.CyclicalLearningRate(
                    1.0,
                    1.0,
                    step_size=1,
                    scale_fn=lambda x: 1.0,
                    scale_mode="cycle",
                )
            )
            recon_bin_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = (
                tfa.optimizers.CyclicalLearningRate(
                    1.0,
                    1.0,
                    step_size=1,
                    scale_fn=lambda x: 1.0,
                    scale_mode="cycle",
                )
            )
            recon_ord_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = (
                tfa.optimizers.CyclicalLearningRate(
                    1.0,
                    1.0,
                    step_size=1,
                    scale_fn=lambda x: 1.0,
                    scale_mode="cycle",
                )
            )
            recon_cat_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = (
                tfa.optimizers.CyclicalLearningRate(
                    1.0,
                    1.0,
                    step_size=1,
                    scale_fn=lambda x: 1.0,
                    scale_mode="cycle",
                )
            )

            class Config:
                arbitrary_types_allowed = True
                smart_union = True

        def __init__(self, config: CoolingRegime.Config, **kwargs):
            super().__init__(**kwargs)
            self.config = config

        def call(self, step):
            cstep = tf.cast(step, self.dtype)
            kld_z_schedule = self.config.kld_z_schedule(cstep)
            recon_schedule = self.config.recon_schedule(cstep)
            recon_reg_schedule = self.config.recon_reg_schedule(cstep)
            recon_bin_schedule = self.config.recon_bin_schedule(cstep)
            recon_ord_schedule = self.config.recon_ord_schedule(cstep)
            recon_cat_schedule = self.config.recon_cat_schedule(cstep)
            return VaeLossNet.InputWeight(
                kld_z_schedule,
                recon_reg_schedule,
                recon_bin_schedule,
                recon_ord_schedule,
                recon_cat_schedule,
            )

    class Config(VaeNet.Config, CoolingRegime.Config):
        pass

    def __init__(self, config: VAE.Config, **kwargs):
        tf.keras.Model.__init__(self, **kwargs)
        self.config = config
        self.network = VaeNet(config, **kwargs)
        self.lossnet = VaeLossNet(latent_eps=1e-6, prefix="loss", **kwargs)
        self.weight_getter = Vae.CoolingRegime(config, dtype=self.dtype)
        AutoencoderModelBaseMixin.__init__(
            self,
            self.weight_getter,
            self.network,
            self.config.get_latent_parser_type(),
            self.config.get_fake_output_getter(),
        )

    @tf.function
    def loss_fn(
        self,
        y_true,
        y_pred: VaeNet.VaeNetOutput,
        weight=VaeLossNet.InputWeight(),
        training=False,
    ) -> VaeLossNet.output:
        y_true = tf.cast(y_true, dtype=self.dtype)
        y_split = self.network.graph_px_g_z.splitter(y_true)

        loss = self.lossnet.Output(
            *[
                tf.reduce_mean(x)
                for x in self.lossnet(
                    self.lossnet.Input.from_output(y_split, y_pred, weight),
                    training,
                )
            ]
        )
        return loss

    @tf.function
    def call(self, x, training=False):
        return self.network(x, training)

    @tf.function
    def latent_sample(self, inputs, y, training=False, samples=1):
        output = self.monte_carlo_estimate(samples, inputs, y, training=training)
        latent = outputs["px_g_z__sample"]
        return latent

    def train_step(self, data, training: bool = False):

        data = data_adapter.expand_1d(data)
        x, y = data
        weights = self.weight_getter(self.optimizer.iterations)

        with backprop.GradientTape() as tape:
            y_pred = self.network(x, training=True)
            losses = self.loss_fn(
                y,
                y_pred,
                weights,
                training=True,
            )
            loss = tf.reduce_mean(losses.loss)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {
            self._output_keys_renamed[k]: v
            for k, v in {
                # **{v.name: v.result() for v in self.metrics}
                **losses._asdict(),
                "kld_z_schedule": weights.lambda_z,
            }.items()
        }

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y = data
        y_pred = self.network(x, training=False)
        losses = self.loss_fn(y, y_pred, VaeLossNet.InputWeight(), training=False)
        loss = tf.reduce_mean(losses.loss)

        return {self._output_keys_renamed[k]: v for k, v in losses._asdict().items()}

    _output_keys_renamed = {
        "kl_z": "losses/kl_z",
        "l_pxgz_reg": "reconstruction/l_pxgz_reg",
        "l_pxgz_bin": "reconstruction/l_pxgz_bin",
        "l_pxgz_ord": "reconstruction/l_pxgz_ord",
        "l_pxgz_cat": "reconstruction/l_pxgz_cat",
        "scaled_l_pxgz": "reconstruction/l_pxgz",
        "scaled_elbo": "losses/scaled_elbo",
        "recon_loss": "losses/recon_loss",
        "loss": "losses/loss",
        "lambda_z": "weight/lambda_z",
        "lambda_reg": "weight/lambda_reg",
        "lambda_bin": "weight/lambda_bin",
        "lambda_ord": "weight/lambda_ord",
        "lambda_cat": "weight/lambda_cat",
        "kld_z_schedule": "weight/lambda_z",
    }

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


class VAE(GenerativeModel):
    class Config(VaeNet.Config):

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
                1.0, 1.0, step_size=1, scale_fn=lambda x: 1.0, scale_mode="cycle"
            )
        )
        recon_bin_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = (
            tfa.optimizers.CyclicalLearningRate(
                1.0, 1.0, step_size=1, scale_fn=lambda x: 1.0, scale_mode="cycle"
            )
        )
        recon_ord_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = (
            tfa.optimizers.CyclicalLearningRate(
                1.0, 1.0, step_size=1, scale_fn=lambda x: 1.0, scale_mode="cycle"
            )
        )
        recon_cat_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = (
            tfa.optimizers.CyclicalLearningRate(
                1.0, 1.0, step_size=1, scale_fn=lambda x: 1.0, scale_mode="cycle"
            )
        )

    def __init__(self, config: VAE.Config, **kwargs):
        super().__init__(**kwargs)
        self.network = VaeNet(config, **kwargs)
        self.lossnet = VaeLossNet(latent_eps=1e-6, prefix="loss", **kwargs)

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
                    self.lossnet.Input.from_vaenet_outputs(y_split, y_pred, weight),
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

        kld_z_schedule = self.config.kld_z_schedule(tf.cast(self.optimizer.iterations, self.dtype))
        recon_schedule = self.config.recon_schedule(tf.cast(self.optimizer.iterations, self.dtype))
        recon_reg_schedule = self.config.recon_reg_schedule(
            tf.cast(self.optimizer.iterations, self.dtype)
        )
        recon_bin_schedule = self.config.recon_bin_schedule(
            tf.cast(self.optimizer.iterations, self.dtype)
        )
        recon_ord_schedule = self.config.recon_ord_schedule(
            tf.cast(self.optimizer.iterations, self.dtype)
        )
        recon_cat_schedule = self.config.recon_cat_schedule(
            tf.cast(self.optimizer.iterations, self.dtype)
        )

        weights = VaeLossNet.InputWeight(
            kld_z_schedule,
            recon_reg_schedule,
            recon_bin_schedule,
            recon_ord_schedule,
            recon_cat_schedule,
        )

        with backprop.GradientTape() as tape:
            y_pred = self.network((x, temp), training=True)
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
                "temp": temp,
                "kld_y_schedule": kld_y_schedule,
                "kld_z_schedule": kld_z_schedule,
            }.items()
        }

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y = data
        y_pred = self.network((x, 1.0), training=False)
        losses = self.loss_fn(y, y_pred, GumbleGmvaeNetLossNet.InputWeight(), training=False)
        loss = tf.reduce_mean(losses.loss)

        return {self._output_keys_renamed[k]: v for k, v in losses._asdict().items()}

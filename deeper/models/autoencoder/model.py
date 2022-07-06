from __future__ import annotations

from collections import namedtuple
from functools import singledispatchmethod as overload
from types import SimpleNamespace
from typing import Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from deeper.models.autoencoder.network import AutoencoderNet
from deeper.models.autoencoder.network_loss import AutoencoderLossNet
from deeper.models.autoencoder.utils import AutoencoderTypeGetter
from deeper.models.generalised_autoencoder.base import AutoencoderModelBaseMixin
from deeper.models.vae.decoder_loss import VaeReconLossNet
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.metrics import categorical_accuracy
from pydantic import BaseModel


class Autoencoder(tf.keras.Model, AutoencoderModelBaseMixin, AutoencoderTypeGetter):
    class CoolingRegime(tf.keras.layers.Layer, AutoencoderTypeGetter):
        class Config(BaseModel):
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
            recon_schedule = self.config.recon_schedule(cstep)
            recon_reg_schedule = self.config.recon_reg_schedule(cstep)
            recon_bin_schedule = self.config.recon_bin_schedule(cstep)
            recon_ord_schedule = self.config.recon_ord_schedule(cstep)
            recon_cat_schedule = self.config.recon_cat_schedule(cstep)
            return AutoencoderLossNet.InputWeight(
                recon_reg_schedule,
                recon_bin_schedule,
                recon_ord_schedule,
                recon_cat_schedule,
            )

    class Config(AutoencoderNet.Config, CoolingRegime.Config):
        pass

    @overload
    def __init__(self, network: VaeNet, config, **kwargs):
        tf.keras.Model.__init__(self, **kwargs)
        self.config = config
        self.network = network
        self.lossnet = AutoencoderLossNet(latent_eps=1e-6, prefix="loss", **kwargs)
        self.weight_getter = Autoencoder.CoolingRegime(config, dtype=self.dtype)
        AutoencoderModelBaseMixin.__init__(
            self,
            self.weight_getter,
            self.network,
            self.config.get_latent_parser_type()(),
            self.config.get_fake_output_getter()(),
        )

    @__init__.register
    def from_config(self, config: BaseModel, **kwargs):
        network = AutoencoderNet(config, **kwargs)
        self.__init__(network, config, **kwargs)

    def loss_fn(
        self,
        y_true,
        y_pred: AutoencoderNet.Output,
        weight=AutoencoderLossNet.InputWeight(),
        training=False,
    ) -> AutoencoderLossNet.output:
        y_true = tf.cast(y_true, dtype=self.dtype)
        y_split = self.network.decoder.splitter(y_true)
        z = self.lossnet(
            AutoencoderLossNet.Input.from_output(y_split, y_pred, weight),
            training,
        )
        loss = self.lossnet.Output(
            VaeReconLossNet.Output(*[tf.reduce_mean(x) for x in z.losses]),
            VaeReconLossNet.Output(*[tf.reduce_mean(x) for x in z.scaled]),
            tf.reduce_mean(z.loss),
        )
        return loss

    @tf.function
    def call(self, x, training=False):
        return self.network(x, training)

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
            reg = tf.reduce_mean(sum(self.losses))
            loss = losses.loss + tf.cast(reg, self.dtype)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {
            "loss": loss,
            **{"metrics/" + v.name: v.result() for v in self.metrics},
        }

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y = data
        y_pred = self.network(x, training=False)
        losses = self.loss_fn(y, y_pred, AutoencoderLossNet.InputWeight(), training=False)
        reg = tf.reduce_mean(sum(self.losses))
        loss = losses.loss + tf.cast(reg, self.dtype)

        return {
            "loss": loss,
            **{"metrics/" + v.name: v.result() for v in self.metrics},
        }

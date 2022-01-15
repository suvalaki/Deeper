from __future__ import annotations
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from typing import Union, Tuple, Sequence, Optional, NamedTuple
from pydantic.dataclasses import dataclass
from pydantic import BaseModel

from deeper.models.gmvae.base import GmvaeModelBase
from deeper.models.gmvae.gmvae_marginalised_categorical.network import (
    StackedGmvaeNet,
)
from deeper.models.gmvae.gmvae_marginalised_categorical.network_loss import (
    StackedGmvaeLossNet,
)
from deeper.models.vae.network_loss import VaeLossNet

from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop

tfk = tf.keras

Model = tfk.Model


class StackedGmvae(GmvaeModelBase):
    class CoolingRegime(GmvaeModelBase.CoolingRegime):
        def call(self, step):
            cstep = tf.cast(step, self.dtype)
            kld_y_schedule = self.config.kld_y_schedule(cstep)
            kld_z_schedule = self.config.kld_z_schedule(cstep)
            recon_schedule = self.config.recon_schedule(cstep)
            recon_reg_schedule = self.config.recon_reg_schedule(cstep)
            recon_bin_schedule = self.config.recon_bin_schedule(cstep)
            recon_ord_schedule = self.config.recon_ord_schedule(cstep)
            recon_cat_schedule = self.config.recon_cat_schedule(cstep)
            return StackedGmvaeLossNet.InputWeight(
                kld_y_schedule,
                kld_z_schedule,
                recon_reg_schedule,
                recon_bin_schedule,
                recon_ord_schedule,
                recon_cat_schedule,
            )

    class Config(GmvaeModelBase.Config, CoolingRegime.Config):
        ...

    def __init__(self, config: StackedGmvae.Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.network = StackedGmvaeNet(config)
        self.lossnet = StackedGmvaeLossNet()
        self.weight_getter = StackedGmvae.CoolingRegime(config, dtype=self.dtype)

    @tf.function
    def loss_fn(
        self,
        y_true,
        y_pred: StackedGmvaeNet.Output,
        weight: StackedGmvaeLossNet.InputWeight,
        training=False,
    ) -> VaeLossNet.output:

        y_true = tf.cast(y_true, dtype=self.dtype)
        y_split = self.network.graph_marginal_autoencoder.graph_px_g_z.splitter(y_true)
        loss = self.lossnet.Output(
            *[
                tf.reduce_mean(x)
                for x in self.lossnet(
                    self.lossnet.Input.from_output(
                        y_split,
                        y_pred,
                        weight,
                    ),
                    training,
                )
            ]
        )

        return loss

    def call(self, data, training: bool = False):
        return self.network(data, training=False).qy_g_x.argmax

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
                "kld_y_schedule": weights.lambda_y,
                "kld_z_schedule": weights.lambda_z,
            }.items()
        }

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y = data
        y_pred = self.network(x, training=False)
        losses = self.loss_fn(y, y_pred, StackedGmvaeLossNet.InputWeight(), training=False)
        loss = tf.reduce_mean(losses.loss)

        return {self._output_keys_renamed[k]: v for k, v in losses._asdict().items()}

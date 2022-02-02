from __future__ import annotations
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from typing import Union, Tuple, Sequence, Optional, NamedTuple
from pydantic.dataclasses import dataclass
from pydantic import BaseModel

from deeper.models.gmvae.base import GmvaeModelBase
from deeper.models.gmvae.gmvae_pure_sampling.network import GumbleGmvaeNet
from deeper.models.gmvae.gmvae_pure_sampling.network_loss import (
    GumbleGmvaeNetLossNet,
)
from deeper.models.vae.network_loss import VaeLossNet

from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop

tfk = tf.keras

Model = tfk.Model


class GumbleGmvae(GmvaeModelBase):
    class CoolingRegime(GmvaeModelBase.CoolingRegime):
        class Config(GmvaeModelBase.CoolingRegime.Config):
            gumble_temperature_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = (
                tf.keras.optimizers.schedules.PolynomialDecay(
                    initial_learning_rate=1.0,
                    decay_steps=10000,
                    end_learning_rate=0.5,
                    power=1.0,
                )
            )

        def call(self, step):
            cstep = tf.cast(step, self.dtype)
            temp = self.config.gumble_temperature_schedule(cstep)
            kld_y_schedule = self.config.kld_y_schedule(cstep)
            kld_z_schedule = self.config.kld_z_schedule(cstep)
            recon_schedule = self.config.recon_schedule(cstep)
            recon_reg_schedule = self.config.recon_reg_schedule(cstep)
            recon_bin_schedule = self.config.recon_bin_schedule(cstep)
            recon_ord_schedule = self.config.recon_ord_schedule(cstep)
            recon_cat_schedule = self.config.recon_cat_schedule(cstep)
            return [
                temp,
                GumbleGmvaeNetLossNet.InputWeight(
                    kld_y_schedule,
                    kld_z_schedule,
                    recon_reg_schedule,
                    recon_bin_schedule,
                    recon_ord_schedule,
                    recon_cat_schedule,
                ),
            ]

    class Config(CoolingRegime.Config, GumbleGmvaeNet.Config):
        ...

    _output_keys_renamed = {
        "temp": "weight/gumble_temperature",
        **GmvaeModelBase._output_keys_renamed,
    }

    def __init__(self, config: GumbleGmvae.Config, **kwargs):
        GmvaeModelBase.__init__(self, config, **kwargs)

    @tf.function
    def loss_fn(
        self,
        y_true,
        y_pred: GumbleGmvaeNet.Output,
        weight: GumbleGmvaeNetLossNet.InputWeight,
        training=False,
    ) -> GumbleGmvaeNetLossNet.output:

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
        return self.network((data, 1.0), training=False).qy_g_x.argmax

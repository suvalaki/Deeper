from __future__ import annotations
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from typing import Union, Tuple, Sequence, Optional, NamedTuple
from pydantic.dataclasses import dataclass
from pydantic import BaseModel

from deeper.models.gmvae.gmvae_marginalised_categorical.network import StackedGmvaeNet
from deeper.models.gmvae.gmvae_marginalised_categorical.network_loss import StackedGmvaeLossNet
from deeper.models.vae.network_loss import VaeLossNet

from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop

tfk = tf.keras

Model = tfk.Model


class Gmvae(Model):

    class Config(StackedGmvaeNet.Config):        
        kld_y_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = tfa.optimizers.CyclicalLearningRate(
            1.0, 1.0, step_size=1, scale_fn=lambda x: 1.0, scale_mode="cycle"
        )
        kld_z_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = tfa.optimizers.CyclicalLearningRate(
            1.0, 1.0, step_size=1, scale_fn=lambda x: 1.0, scale_mode="cycle"
        )
        recon_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = tfa.optimizers.CyclicalLearningRate(
            1.0, 1.0, step_size=1, scale_fn=lambda x: 1.0, scale_mode="cycle"
        )
        recon_reg_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = tfa.optimizers.CyclicalLearningRate(
            1.0, 1.0, step_size=1, scale_fn=lambda x: 1.0, scale_mode="cycle"
        )
        recon_bin_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = tfa.optimizers.CyclicalLearningRate(
            1.0, 1.0, step_size=1, scale_fn=lambda x: 1.0, scale_mode="cycle"
        )
        recon_ord_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = tfa.optimizers.CyclicalLearningRate(
            1.0, 1.0, step_size=1, scale_fn=lambda x: 1.0, scale_mode="cycle"
        )
        recon_cat_schedule: tf.keras.optimizers.schedules.LearningRateSchedule = tfa.optimizers.CyclicalLearningRate(
            1.0, 1.0, step_size=1, scale_fn=lambda x: 1.0, scale_mode="cycle"
        )


    def __init__(
        self, config: Gmvae.Config, **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config
        self.network = StackedGmvaeNet(config)
        self.lossnet = StackedGmvaeLossNet()

    @tf.function
    def loss_fn(
        self, y_true, y_pred: StackedGmvaeNet.Output, weight: StackedGmvaeLossNet.InputWeight, training=False
    ) -> VaeLossNet.output:

        y_true = tf.cast(y_true, dtype=self.dtype)
        y_split = self.network.graph_marginal_autoencoder.graph_px_g_z.splitter(y_true)
        loss = self.lossnet.Output(
            *[
                tf.reduce_mean(x)
                for x in self.lossnet(
                    self.lossnet.Input.from_StackedGmvaeNet_output(
                        y_split,
                        y_pred,
                        weight,
                    ),
                    training,
                )
            ]
        )

        return loss

    def call(self, data, training:bool = False):
        return self.network(data, training=False).qy_g_x.argmax

    def train_step(self, data, training:bool = False):

        data = data_adapter.expand_1d(data)
        x, y = data

        kld_y_schedule = self.config.kld_y_schedule(
            tf.cast(self.optimizer.iterations, self.dtype)
        )
        kld_z_schedule = self.config.kld_z_schedule(
            tf.cast(self.optimizer.iterations, self.dtype)
        )
        recon_schedule = self.config.recon_schedule(
            tf.cast(self.optimizer.iterations, self.dtype)
        )
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

        weights = StackedGmvaeLossNet.InputWeight(
            kld_y_schedule,
            kld_z_schedule,
            recon_reg_schedule, 
            recon_bin_schedule, 
            recon_ord_schedule, 
            recon_cat_schedule, 
        )

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
            #**{v.name: v.result() for v in self.metrics}
            **losses._asdict(), 
            "kld_y_schedule":kld_y_schedule, 
            "kld_z_schedule": kld_z_schedule,
        }

    def test_step(self, data ):
        data = data_adapter.expand_1d(data)
        x, y = data
        y_pred = self.network(x, training=False)
        losses = self.loss_fn(
            y, y_pred, GumbleGmvaeNetLossNet.InputWeight(), training=False
        )
        loss = tf.reduce_mean(losses.total_loss)

        return {
            **losses._asdict(), 
            #**{v.name: v.result() for v in self.metrics},
        }

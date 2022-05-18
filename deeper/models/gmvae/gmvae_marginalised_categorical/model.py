from __future__ import annotations
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from typing import Union, Tuple, Sequence, Optional, NamedTuple
from pydantic.dataclasses import dataclass
from pydantic import BaseModel
from functools import singledispatchmethod as overload

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
        class Config(GmvaeModelBase.CoolingRegime.Config):
            ...

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

    class Config(CoolingRegime.Config, StackedGmvaeNet.Config, GmvaeModelBase.Config):
        ...

    @overload
    def __init__(self, network: StackedGmvaeNet, config: StackedGmvaeNet.Config, **kwargs):
        GmvaeModelBase.__init__(self, network, config, **kwargs)

    @__init__.register
    def from_config(self, config: BaseModel, **kwargs):
        GmvaeModelBase.from_config(self, config, **kwargs)

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

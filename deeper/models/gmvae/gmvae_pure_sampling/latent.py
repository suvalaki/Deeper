from __future__ import annotations

import tensorflow as tf
from deeper.models.generalised_autoencoder.base import LatentParser
from deeper.models.gmvae.gmvae_pure_sampling import GumbleGmvaeNet


class GumbleGmvaeLatentParser(LatentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x: GumbleGmvaeNet.Output, training=False):
        return x.marginal.qz_g_xy.sample

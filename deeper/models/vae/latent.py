from __future__ import annotations

import tensorflow as tf
from deeper.models.generalised_autoencoder.base import LatentParser
from deeper.models.vae import VaeNet


class VaeLatentParser(LatentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self, x: VaeNet.Output, training=False):
        return x.qz_g_x.sample

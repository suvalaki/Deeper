from __future__ import annotations

import tensorflow as tf
from deeper.models.autoencoder.network import AutoencoderNet
from deeper.models.generalised_autoencoder.base import LatentParser


class AutoencoderLatentParser(LatentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self, x: AutoencoderNet.Output, training=False):
        return x.latent

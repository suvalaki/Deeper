from __future__ import annotations

import tensorflow as tf
from deeper.models.generalised_autoencoder.base import LatentParser
from deeper.models.gmvae.gmvae_marginalised_categorical import StackedGmvaeNet


class StackedGmvaeLatentParser(LatentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x: StackedGmvaeNet.Output, training=False):
        return tf.reduce_sum(
            x.qy_g_x.probs[:, None, :] * tf.stack([m.qz_g_xy.sample for m in x.marginals], axis=-1),
            axis=-1,
        )

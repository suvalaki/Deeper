import tensorflow as tf


import pytest
import unittest
import numpy as np
import os

from typing import NamedTuple
from deeper.utils.data.dummy import generate_dummy_dataset_alltypes
from deeper.models.vae.network import VaeNet, MultipleObjectiveDimensions
from deeper.models.vae.network_loss import VaeLossNet

from deeper.models.gan.network import GanNet
from deeper.models.gan.descriminator import DescriminatorNet
from deeper.models.gan.network_loss import GanLossNet


N_ROWS = 25
DIM_REG = 10
DIM_BOOL = 15
DIM_ORD = (5,)
DIM_CAT = (10, 8)
DIM_X = DIM_REG + DIM_BOOL + sum(DIM_ORD) + sum(DIM_CAT)
EMB_DIM = 10
LAT_DIM = 5
NCATS = 5

desciminatorConfig = DescriminatorNet.Config(embedding_dimensions=[EMB_DIM])

vaeConfig = VaeNet.Config(
    input_dimensions=MultipleObjectiveDimensions(
        regression=DIM_REG,
        boolean=DIM_BOOL,
        ordinal=DIM_ORD,
        categorical=DIM_CAT,
    ),
    output_dimensions=MultipleObjectiveDimensions(
        regression=DIM_REG,
        boolean=DIM_BOOL,
        ordinal=DIM_ORD,
        categorical=DIM_CAT,
    ),
    encoder_embedding_dimensions=[EMB_DIM],
    decoder_embedding_dimensions=[EMB_DIM],
    latent_dim=LAT_DIM,
)


config = GanNet.Config(descriminator=desciminatorConfig, generator=vaeConfig)


class TestGanLossNet(unittest.TestCase):
    def setUp(self):
        state = np.random.RandomState(0)
        X = generate_dummy_dataset_alltypes(
            state, N_ROWS, DIM_REG, DIM_BOOL, DIM_ORD, DIM_CAT
        ).X
        temps = state.random((N_ROWS, 1))

        self.data = X
        self.network = GanNet(config, dtype=tf.dtypes.float64)
        self.lossnet = GanLossNet()

    def test_outputShapes(self):
        x = y_true = self.data
        pred_generative = self.network.call_generative(x)
        pred_descriminative = self.network.call_descriminative(x, x)

        fool_descrim = self.lossnet.call_fool_descriminator(
            y_true, pred_generative
        )
        tune_descrim = self.lossnet.call_tune_descriminator(
            y_true, pred_descriminative
        )

        self.assertEqual(fool_descrim.shape, (x.shape[0], 1))
        self.assertEqual(tune_descrim.shape, (x.shape[0], 1))


if __name__ == "__main__":
    # USE CPU ONLY
    # tf.config.set_visible_devices([], "GPU")
    unittest.main()

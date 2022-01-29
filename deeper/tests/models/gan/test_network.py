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


class TestGanNet(unittest.TestCase):
    def setUp(self):
        state = np.random.RandomState(0)
        X = generate_dummy_dataset_alltypes(
            state, N_ROWS, DIM_REG, DIM_BOOL, DIM_ORD, DIM_CAT
        ).X
        temps = state.random((N_ROWS, 1))

        self.data = X
        self.network = GanNet(config, dtype=tf.dtypes.float64)

    def test_outputShapes(self):
        pred_generative = self.network.call_generative(self.data[0:1, :])
        pred_descriminative = self.network.call_descriminative(
            self.data[0:1, :], self.data[0:1, :]
        )
        # Latent layers
        self.assertEqual(pred_generative.fake_descriminant.prob.shape, (1, 1))
        self.assertEqual(
            pred_descriminative.fake_descriminant.prob.shape, (1, 1)
        )


if __name__ == "__main__":
    # USE CPU ONLY
    # tf.config.set_visible_devices([], "GPU")
    unittest.main()

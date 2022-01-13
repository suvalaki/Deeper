import tensorflow as tf

import pytest
import unittest
import numpy as np
import os

from typing import NamedTuple
from deeper.utils.data.dummy import generate_dummy_dataset_alltypes
from deeper.models.gmvae.gmvae_marginalised_categorical.network import (
    StackedGmvaeNet,
)
from deeper.models.gmvae.gmvae_marginalised_categorical.network_loss import (
    StackedGmvaeLossNet,
)
from deeper.models.gmvae.gmvae_marginalised_categorical.latent import StackedGmvaeLatentParser
from deeper.models.vae import MultipleObjectiveDimensions
from deeper.models.vae.network_loss import VaeLossNet


N_ROWS = 25
DIM_REG = 10
DIM_BOOL = 15
DIM_ORD = (5,)
DIM_CAT = (10, 8)
DIM_X = DIM_REG + DIM_BOOL + sum(DIM_ORD) + sum(DIM_CAT)
EMB_DIM = 10
LAT_DIM = 5
NCATS = 5

config = StackedGmvaeNet.Config(
    components=2,
    cat_embedding_dimensions=[EMB_DIM],
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


class TestGumbleGmVae(unittest.TestCase):
    def setUp(self):
        state = np.random.RandomState(0)
        X = generate_dummy_dataset_alltypes(state, N_ROWS, DIM_REG, DIM_BOOL, DIM_ORD, DIM_CAT).X
        temps = state.random((N_ROWS, 1))

        self.data = (X, temps)
        self.network = StackedGmvaeNet(config)
        self.parser = StackedGmvaeLatentParser()

    def test_outputShapes(self):
        pred = self.network(self.data[0])
        latent = self.parser(pred)
        self.assertEqual(latent.shape, (N_ROWS, LAT_DIM))


if __name__ == "__main__":
    # USE CPU ONLY
    tf.config.set_visible_devices([], "GPU")
    unittest.main()

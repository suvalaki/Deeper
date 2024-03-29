import tensorflow as tf

import pytest
import unittest
import numpy as np
import os

from typing import NamedTuple
from deeper.models.vae import MultipleObjectiveDimensions
from deeper.utils.data.dummy import generate_dummy_dataset_alltypes
from deeper.models.gmvae.gmvae_marginalised_categorical import StackedGmvae


N_ROWS = 25
DIM_REG = 10
DIM_BOOL = 15
DIM_ORD = (5,)
DIM_CAT = (10, 8)
DIM_X = DIM_REG + DIM_BOOL + sum(DIM_ORD) + sum(DIM_CAT)
EMB_DIM = 10
LAT_DIM = 5
NCATS = 5

config = StackedGmvae.Config(
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


class TestStackedGmVae(unittest.TestCase):
    def setUp(self):
        state = np.random.RandomState(0)
        X = generate_dummy_dataset_alltypes(state, N_ROWS, DIM_REG, DIM_BOOL, DIM_ORD, DIM_CAT).X
        temps = state.random((N_ROWS, 1))

        self.data = X
        self.model = StackedGmvae(config)
        self.model.compile()

    def test_outputShapes(self):
        pred = self.model(self.data)
        self.assertEqual(pred.shape, (N_ROWS,))

    def test_train_step(self):
        vals = self.model.train_step([self.data, self.data])

    def test_test_step(self):
        vals = self.model.test_step([self.data, self.data])


if __name__ == "__main__":
    # USE CPU ONLY
    tf.config.set_visible_devices([], "GPU")
    unittest.main()

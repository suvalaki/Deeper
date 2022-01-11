import tensorflow as tf
# USE CPU ONLY
tf.config.set_visible_devices([], 'GPU')

import pytest
import unittest
import numpy as np
import os

from typing import NamedTuple
from deeper.utils.data.dummy import generate_dummy_dataset_alltypes
from deeper.models.gmvae.gmvae_pure_sampling import GumbleGmvae


N_ROWS = 25
DIM_REG = 10
DIM_BOOL = 15
DIM_ORD = (5,)
DIM_CAT = (10, 8)
DIM_X = DIM_REG + DIM_BOOL + sum(DIM_ORD) + sum(DIM_CAT)
EMB_DIM = 10
LAT_DIM = 5
NCATS=5

config = GumbleGmvae.Config(
    components = 2,
    cat_embedding_dimensions = [EMB_DIM],
    input_regression_dimension = DIM_REG,
    input_boolean_dimension = DIM_BOOL,
    input_ordinal_dimension = DIM_ORD,
    input_categorical_dimension = DIM_CAT,
    output_regression_dimension = DIM_REG,
    output_boolean_dimension = DIM_BOOL,
    output_ordinal_dimension = DIM_ORD,
    output_categorical_dimension = DIM_CAT,
    encoder_embedding_dimensions = [EMB_DIM],
    decoder_embedding_dimensions = [EMB_DIM],
    latent_dim = LAT_DIM
)


class TestGumbleGmVae(unittest.TestCase):
    def setUp(self):
        state = np.random.RandomState(0)
        X = generate_dummy_dataset_alltypes(
            state, N_ROWS, DIM_REG, DIM_BOOL, DIM_ORD, DIM_CAT).X
        temps = state.random((N_ROWS, 1))

        self.data = X
        self.model = GumbleGmvae(config)
        self.model.compile()

    def test_outputShapes(self):
        pred = self.model(self.data)
        self.assertEqual(pred.shape, (N_ROWS,))

    def test_train_step(self):
        vals = self.model.train_step([self.data, self.data])

    def test_test_step(self):
        vals = self.model.test_step([self.data, self.data])


if __name__ == "__main__":
    unittest.main()

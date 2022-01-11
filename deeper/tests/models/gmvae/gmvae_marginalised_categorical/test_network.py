import tensorflow as tf
# USE CPU ONLY
tf.config.set_visible_devices([], 'GPU')

import pytest
import unittest
import numpy as np
import os

from typing import NamedTuple
from deeper.utils.data.dummy import generate_dummy_dataset_alltypes
from deeper.models.gmvae.gmvae_marginalised_categorical.network import StackedGmvaeNet
from deeper.models.gmvae.gmvae_marginalised_categorical.network_loss import StackedGmvaeLossNet
from deeper.models.vae.network_loss import VaeLossNet


N_ROWS = 25
DIM_REG = 10
DIM_BOOL = 15
DIM_ORD = (5,)
DIM_CAT = (10, 8)
DIM_X = DIM_REG + DIM_BOOL + sum(DIM_ORD) + sum(DIM_CAT)
EMB_DIM = 10
LAT_DIM = 5
NCATS=5

config = StackedGmvaeNet.Config(
    components = 2,
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

        self.data = (X, temps)
        self.network = StackedGmvaeNet(config)

    def test_outputShapes(self):
        pred = self.network(self.data[0][0:1,:])
        # Latent layers

    def test_loss(self):

        lossnet = StackedGmvaeLossNet()
        pred = self.network(self.data[0][0:1,:])
        y_true = self.network.graph_marginal_autoencoder.graph_px_g_z.splitter(
            self.data[0][0:1,:])
        inputs = StackedGmvaeLossNet.Input.from_StackedGmvaeNet_output(
            y_true = y_true, 
            model_output = pred, 
            weights = StackedGmvaeLossNet.InputWeight(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )
        losses = lossnet(inputs)
        print(losses)

        from pprint import pp
        getShape = lambda x: [v.shape for v in x] if isinstance(x, list) else x.shape
        pp({k:v for k,v in losses._asdict().items()}, depth=6, indent=4)




if __name__ == "__main__":
    unittest.main()
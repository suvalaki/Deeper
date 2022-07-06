import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import pytest
import unittest
import numpy as np
import os

from typing import NamedTuple
from deeper.utils.data.dummy import generate_dummy_dataset_alltypes
from deeper.models.autoencoder.network import AutoencoderNet, MultipleObjectiveDimensions
from deeper.models.autoencoder.network_loss import AutoencoderLossNet


N_ROWS = 25
DIM_REG = 10
DIM_BOOL = 15
DIM_ORD = (5,)
DIM_CAT = (10, 8)
DIM_X = DIM_REG + DIM_BOOL + sum(DIM_ORD) + sum(DIM_CAT)
EMB_DIM = 10
LAT_DIM = 5
NCATS = 5

config = AutoencoderNet.Config(
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


class TestAutoencoderNet(unittest.TestCase):
    def setUp(self):
        state = np.random.RandomState(0)
        X = generate_dummy_dataset_alltypes(state, N_ROWS, DIM_REG, DIM_BOOL, DIM_ORD, DIM_CAT).X
        temps = state.random((N_ROWS, 1))

        self.data = X
        self.network = AutoencoderNet(config, dtype=tf.dtypes.float64)

    def test_getNetworkFromConfig(self):
        net = config.get_network_type()(config)
        lossnet = config.get_lossnet_type()(prefix="loss")
        pred = net(self.data[0:1, :])
        y_true = net.decoder.splitter(self.data[0:1, :])
        inputs = config.get_lossnet_type().Input.from_output(
            y_true=y_true,
            model_output=pred,
            weights=AutoencoderLossNet.InputWeight(1.0, 1.0, 1.0, 1.0),
        )
        losses = lossnet(inputs)

    def test_outputShapes(self):
        pred = self.network(self.data[0:1, :])
        # Latent layers

    def test_loss(self):

        lossnet = AutoencoderLossNet(prefix="loss")
        pred = self.network(self.data[0:1, :])
        y_true = self.network.decoder.splitter(self.data[0:1, :])
        inputs = AutoencoderLossNet.Input.from_output(
            y_true=y_true,
            model_output=pred,
            weights=AutoencoderLossNet.InputWeight(1.0, 1.0, 1.0, 1.0),
        )
        losses = lossnet(inputs)


if __name__ == "__main__":
    # USE CPU ONLY
    # tf.config.set_visible_devices([], "GPU")
    unittest.main()

import tensorflow as tf

import pytest
import unittest
import numpy as np
import os

from typing import NamedTuple
from deeper.utils.data.dummy import generate_dummy_dataset_alltypes

from deeper.models.vae import (
    MultipleObjectiveDimensions,
    VaeNet,
    VaeLossNet,
    VaeLatentParser,
)
from deeper.models.gmvae.gmvae_marginalised_categorical import (
    StackedGmvaeNet,
    StackedGmvaeLossNet,
    StackedGmvaeLatentParser,
)
from deeper.models.gmvae.gmvae_pure_sampling import (
    GumbleGmvaeNet,
    GumbleGmvaeNetLossNet,
    GumbleGmvaeLatentParser,
)
from deeper.models.generalised_autoencoder.network import (
    GeneralisedAutoencoderNet,
)


N_ROWS = 25
DIM_REG = 10
DIM_BOOL = 15
DIM_ORD = (5,)
DIM_CAT = (10, 8)
DIM_X = DIM_REG + DIM_BOOL + sum(DIM_ORD) + sum(DIM_CAT)
EMB_DIM = 10
LAT_DIM = 5
NCATS = 5

config0 = VaeNet.Config(
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

config1 = StackedGmvaeNet.Config(
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

config2 = GumbleGmvaeNet.Config(
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

CONFIGS = [config0, config1, config2]
EXPECTED_TYPES = [
    (VaeNet, VaeLossNet, VaeLatentParser),
    (StackedGmvaeNet, StackedGmvaeLossNet, StackedGmvaeLatentParser),
    (GumbleGmvaeNet, GumbleGmvaeNetLossNet, GumbleGmvaeLatentParser),
]


class TestGeneralisedAutoencoderNet(unittest.TestCase):
    def setUp(self):
        state = np.random.RandomState(0)
        X = generate_dummy_dataset_alltypes(state, N_ROWS, DIM_REG, DIM_BOOL, DIM_ORD, DIM_CAT).X
        temps = state.random((N_ROWS, 1))

        self.data = ((X, X), (X, X), ([X, temps], X))
        self.networks = [GeneralisedAutoencoderNet(c) for c in CONFIGS]

    def test_layerTypes(self):
        for net, config, exp_type in zip(self.networks, CONFIGS, EXPECTED_TYPES):
            self.assertEqual(type(net.network), exp_type[0])
            self.assertEqual(type(net.lossnet), exp_type[1])
            self.assertEqual(type(net.latent_parser), exp_type[2])

    def test_call(self):
        for net, config, exp_type, (x, y) in zip(self.networks, CONFIGS, EXPECTED_TYPES, self.data):
            net(x, y)


if __name__ == "__main__":
    # USE CPU ONLY
    tf.config.set_visible_devices([], "GPU")
    unittest.main()

import tensorflow as tf


import pytest
import unittest
import numpy as np
import os

from typing import NamedTuple
from deeper.utils.data.dummy import generate_dummy_dataset_alltypes
from deeper.models.vae.network import VaeNet, MultipleObjectiveDimensions
from deeper.models.vae.network_loss import VaeLossNet
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

from deeper.models.gan.network import GanNet
from deeper.models.gan.network_loss import GanLossNet
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

CONFIGS = [
    GanNet.Config(descriminator=desciminatorConfig, generator=config0),
    GanNet.Config(descriminator=desciminatorConfig, generator=config1),
    GanNet.Config(descriminator=desciminatorConfig, generator=config2),
]
EXPECTED_TYPES = [
    (VaeNet, VaeLossNet, VaeLatentParser),
    (StackedGmvaeNet, StackedGmvaeLossNet, StackedGmvaeLatentParser),
    (GumbleGmvaeNet, GumbleGmvaeNetLossNet, GumbleGmvaeLatentParser),
]


class TestGanLossNet(unittest.TestCase):
    def setUp(self):
        state = np.random.RandomState(0)
        X = generate_dummy_dataset_alltypes(state, N_ROWS, DIM_REG, DIM_BOOL, DIM_ORD, DIM_CAT).X
        temps = state.random((N_ROWS, 1))

        self.data = ((X, X), (X, X), ([X, temps], X))
        self.networks = [GanNet(c) for c in CONFIGS]
        self.lossnet = GanLossNet()

    def test_outputShapes(self):
        for net, config, exp_type, (x, y) in zip(self.networks, CONFIGS, EXPECTED_TYPES, self.data):
            pred_generative = net.call_generative(x)
            pred_descriminative = net.call_descriminative(x, y)

            fool_descrim = self.lossnet.call_fool_descriminator(y, pred_generative)
            tune_descrim = self.lossnet.call_tune_descriminator(y, pred_descriminative)

            self.assertEqual(fool_descrim.shape, (N_ROWS, 1))
            self.assertEqual(tune_descrim.shape, (N_ROWS, 1))


if __name__ == "__main__":
    # USE CPU ONLY
    # tf.config.set_visible_devices([], "GPU")
    unittest.main()

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
    Vae,
    VaeNet,
    VaeLossNet,
    VaeLatentParser,
)
from deeper.models.gmvae.gmvae_marginalised_categorical import (
    StackedGmvae,
    StackedGmvaeNet,
    StackedGmvaeLossNet,
    StackedGmvaeLatentParser,
)
from deeper.models.gmvae.gmvae_pure_sampling import (
    GumbleGmvae,
    GumbleGmvaeNet,
    GumbleGmvaeNetLossNet,
    GumbleGmvaeLatentParser,
)

from deeper.models.gan.network import GanNet
from deeper.models.gan.descriminator import DescriminatorNet

from deeper.models.adversarial_autoencoder.network import (
    AdversarialAuoencoderNet,
)
from deeper.models.adversarial_autoencoder.network_loss import (
    AdverasrialAutoencoderLossNet,
)
from deeper.models.adversarial_autoencoder.model import (
    AdversarialAutoencoder,
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

desciminatorConfig = DescriminatorNet.Config(embedding_dimensions=[EMB_DIM])

vaeConfig = Vae.Config(
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

config0 = Vae.Config(
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

config1 = StackedGmvae.Config(
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

config2 = GumbleGmvae.Config(
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
    AdversarialAutoencoder.Config(descriminator=desciminatorConfig, generator=config0),
    AdversarialAutoencoder.Config(descriminator=desciminatorConfig, generator=config1),
    AdversarialAutoencoder.Config(descriminator=desciminatorConfig, generator=config2),
]
EXPECTED_TYPES = [
    (VaeNet, VaeLossNet, VaeLatentParser),
    (StackedGmvaeNet, StackedGmvaeLossNet, StackedGmvaeLatentParser),
    (GumbleGmvaeNet, GumbleGmvaeNetLossNet, GumbleGmvaeLatentParser),
]


class TestAdversarialAutoencoderModel(unittest.TestCase):
    def setUp(self):
        state = np.random.RandomState(0)
        X = generate_dummy_dataset_alltypes(state, N_ROWS, DIM_REG, DIM_BOOL, DIM_ORD, DIM_CAT).X
        temps = state.random((N_ROWS, 1))

        self.data = ((X, X), (X, X), (X, X), ([X, temps], X))
        self.models = [AdversarialAutoencoder(c) for c in CONFIGS]
        for m in self.models:
            m.compile()

    def test_outputShapes(self):
        for model, config, exp_type, (x, y) in zip(self.models, CONFIGS, EXPECTED_TYPES, self.data):
            pred = model(x)

    def test_train_step(self):
        for model, config, exp_type, (x, y) in zip(self.models, CONFIGS, EXPECTED_TYPES, self.data):
            vals = model.train_step((x, y))

    def test_test_step(self):
        for model, config, exp_type, (x, y) in zip(self.models, CONFIGS, EXPECTED_TYPES, self.data):
            vals = model.test_step((x, y))


if __name__ == "__main__":
    # USE CPU ONLY
    # tf.config.set_visible_devices([], "GPU")
    unittest.main()

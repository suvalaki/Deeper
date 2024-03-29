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
from deeper.models.gan.descriminator import DescriminatorNet

from deeper.models.adversarial_autoencoder.network import (
    AdversarialAutoencoderNet,
)
from deeper.models.adversarial_autoencoder.network_loss import (
    AdverasrialAutoencoderLossNet,
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
    AdversarialAutoencoderNet.Config(descriminator=desciminatorConfig, generator=config0),
    AdversarialAutoencoderNet.Config(descriminator=desciminatorConfig, generator=config1),
    AdversarialAutoencoderNet.Config(descriminator=desciminatorConfig, generator=config2),
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
        self.networks = [AdversarialAutoencoderNet(c) for c in CONFIGS]
        self.lossnets = [
            AdverasrialAutoencoderLossNet(c, net) for net, c in zip(self.networks, CONFIGS)
        ]

    def test_outputShapes(self):

        for net, lossnet, config, exp_type, (x, y) in zip(
            self.networks, self.lossnets, CONFIGS, EXPECTED_TYPES, self.data
        ):
            pred_generative = net.call_generative(x)
            pred_descriminative = net.call_descriminative(x, y)
            y_pred = net.call(x, y)
            lossnet_input = lossnet.Input.from_output(
                net,
                lossnet.generator_lossnet,
                y,
                y_pred,
                lossnet.generator_lossnet.InputWeight(),
            )
            loss_out = lossnet(lossnet_input)

            self.assertEqual(loss_out.generative.shape, (N_ROWS, 1))
            self.assertEqual(loss_out.descriminative.shape, (N_ROWS, 1))
            # Loss outs are flat onto the batch size
            self.assertEqual(loss_out.reconstruciton.shape, (N_ROWS,))


if __name__ == "__main__":
    # USE CPU ONLY
    # tf.config.set_visible_devices([], "GPU")
    unittest.main()

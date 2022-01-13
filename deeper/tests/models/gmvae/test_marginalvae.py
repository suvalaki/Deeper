import tensorflow as tf

import pytest
import unittest
import numpy as np
import os

from deeper.models.vae import MultipleObjectiveDimensions
from deeper.models.vae.network_loss import VaeLossNet
from deeper.models.gmvae.marginalvae import MarginalGmVaeNet
from typing import NamedTuple
from deeper.utils.data.dummy import generate_dummy_dataset_alltypes
from deeper.models.gmvae.marginalvae_loss import MarginalGmVaeLossNet
from deeper.models.vae.decoder_loss import VaeReconLossNet
from deeper.models.vae.utils import SplitCovariates


N_ROWS = 25
DIM_REG = 10
DIM_BOOL = 15
DIM_ORD = (5,)
DIM_CAT = (10, 8)
DIM_X = DIM_REG + DIM_BOOL + sum(DIM_ORD) + sum(DIM_CAT)
EMB_DIM = 10
LAT_DIM = 5
NCATS = 5

config = MarginalGmVaeNet.Config(
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


class Inputs(NamedTuple):
    X: tf.Tensor
    y: tf.Tensor


class TestMarginalVae(unittest.TestCase):
    def setUp(self):
        state = np.random.RandomState(0)
        X_all = generate_dummy_dataset_alltypes(
            state, N_ROWS, DIM_REG, DIM_BOOL, DIM_ORD, DIM_CAT
        )
        X = X_all.X
        # categories
        y = (state.random((N_ROWS, NCATS)) > 0.5).astype(float)

        self.data = (X, y, X_all)
        self.network = MarginalGmVaeNet(config)

    def test_outputShapes(self):

        pred = self.network([self.data[0][0:1, :], self.data[1][0:1, :]])

        # Latent layers
        self.assertEqual(pred.qz_g_xy.sample.shape[-1], LAT_DIM)
        self.assertEqual(pred.pz_g_y.sample.shape[-1], LAT_DIM)
        # # Reconstruction
        self.assertEqual(pred.px_g_zy.hidden_logits.shape[-1], DIM_X)
        self.assertEqual(pred.px_g_zy.regression.shape[-1], DIM_REG)
        self.assertEqual(pred.px_g_zy.logits_binary.shape[-1], DIM_BOOL)
        self.assertEqual(pred.px_g_zy.binary.shape[-1], DIM_BOOL)
        self.assertEqual(
            pred.px_g_zy.logits_ordinal_groups_concat.shape[-1], sum(DIM_ORD)
        )
        self.assertEqual(
            tuple([x.shape[-1] for x in pred.px_g_zy.logits_ordinal_groups]),
            DIM_ORD,
        )
        self.assertEqual(
            pred.px_g_zy.ord_groups_concat.shape[-1], sum(DIM_ORD)
        )
        self.assertEqual(
            tuple([x.shape[-1] for x in pred.px_g_zy.ord_groups]), DIM_ORD
        )
        self.assertEqual(
            pred.px_g_zy.logits_categorical_groups_concat.shape[-1],
            sum(DIM_CAT),
        )
        self.assertEqual(
            tuple(
                [x.shape[-1] for x in pred.px_g_zy.logits_categorical_groups]
            ),
            DIM_CAT,
        )
        self.assertEqual(
            pred.px_g_zy.categorical_groups_concat.shape[-1], sum(DIM_CAT)
        )
        self.assertEqual(
            tuple([x.shape[-1] for x in pred.px_g_zy.categorical_groups]),
            DIM_CAT,
        )

    def test_loss(self):

        lossnet = MarginalGmVaeLossNet()
        pred = self.network([self.data[0][0:1, :], self.data[1][0:1, :]])
        y_true = self.network.graph_px_g_z.splitter(self.data[0][0:1, :])
        inputs = MarginalGmVaeLossNet.Input.from_MarginalGmVae_output(
            y_true=y_true,
            model_output=pred,
            weights=VaeLossNet.InputWeight(1.0, 1.0, 1.0, 1.0, 1.0),
        )

        from pprint import pp

        # getShape = lambda x: [v.shape for v in x] if isinstance(x, list) else x.shape
        # pp({k:getShape(v) for k,v in inputs.y_true._asdict().items()}, depth=6, indent=4)
        # print("=====================")
        # pp({k:getShape(v) for k,v in inputs.y_pred._asdict().items()}, depth=6, indent=4)

        losses = lossnet(inputs, False)


if __name__ == "__main__":
    # USE CPU ONLY
    tf.config.set_visible_devices([], "GPU")
    unittest.main()

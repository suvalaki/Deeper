import tensorflow as tf

import pytest
import unittest
import numpy as np
import os

from typing import NamedTuple
from deeper.utils.data.dummy import generate_dummy_dataset_alltypes
from deeper.models.autoencoder.model import Autoencoder
from deeper.models.autoencoder.network import MultipleObjectiveDimensions


N_ROWS = 25
DIM_REG = 10
DIM_BOOL = 15
DIM_ORD = (5,)
DIM_CAT = (10, 8)
DIM_X = DIM_REG + DIM_BOOL + sum(DIM_ORD) + sum(DIM_CAT)
EMB_DIM = 10
LAT_DIM = 5
NCATS = 5

config = Autoencoder.Config(
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


class TestAutoencoderModel(unittest.TestCase):
    def setUp(self):
        state = np.random.RandomState(0)
        X = generate_dummy_dataset_alltypes(state, N_ROWS, DIM_REG, DIM_BOOL, DIM_ORD, DIM_CAT).X

        self.data = X
        self.model = Autoencoder(config)
        self.model.compile()

    def test_outputShapes(self):
        pred = self.model(self.data)

        # Latent layers
        self.assertEqual(pred.latent.shape[-1], LAT_DIM)
        # # Reconstruction
        self.assertEqual(
            tuple([x.shape[-1] for x in pred.reconstruction.logits_ordinal_groups]),
            DIM_ORD,
        )
        self.assertEqual(pred.reconstruction.ord_groups_concat.shape[-1], sum(DIM_ORD))
        self.assertEqual(tuple([x.shape[-1] for x in pred.reconstruction.ord_groups]), DIM_ORD)
        self.assertEqual(
            pred.reconstruction.logits_categorical_groups_concat.shape[-1],
            sum(DIM_CAT),
        )
        self.assertEqual(
            tuple([x.shape[-1] for x in pred.reconstruction.logits_categorical_groups]),
            DIM_CAT,
        )
        self.assertEqual(pred.reconstruction.categorical_groups_concat.shape[-1], sum(DIM_CAT))
        self.assertEqual(
            tuple([x.shape[-1] for x in pred.reconstruction.categorical_groups]),
            DIM_CAT,
        )

    def test_train_step(self):
        vals = self.model.train_step([self.data, self.data])

    def test_test_step(self):
        vals = self.model.test_step([self.data, self.data])


if __name__ == "__main__":
    # USE CPU ONLY
    unittest.main()

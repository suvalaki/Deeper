import pytest
import unittest
import numpy as np
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)

from deeper.models.vae.model import VAE

class TestVAE(unittest.TestCase):
    def test___init__(self):

        self.assertTrue(True)

        # Constructs with all dimensions provided
        # activations and initializers using strings
        model = VAE(
            input_regression_dimension=5,
            input_boolean_dimension=6,
            input_cateorical_dimension=(2, 3),
            ouput_regression_dimension=8,
            output_boolean_dimension=9,
            output_categorical_dimension=(4, 5),
            latent_dim=15,
        )

        self.assertEqual(model.input_dimenson, 5 + 6 + 2 + 3)
        self.assertEqual(model.input_categorical_dimensions, 6)

    def test_regression_only(self):
        dummy_data = np.random.normal(0, 1, (100, 10))
        model = VAE(
            input_regression_dimension=10,
            input_boolean_dimension=0,
            input_ordinal_dimension=0,
            input_categorical_dimension=0,
            output_regression_dimension=10,
            output_boolean_dimension=0,
            output_ordinal_dimension=0,
            output_categorical_dimension=0,
            encoder_embedding_dimensions=(10,),
            decoder_embedding_dimensions=(10,),
            latent_dim=5,
        )
        model.call((dummy_data, dummy_data))
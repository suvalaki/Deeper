import pytest
import unittest


from deeper.models.vae import VAE


class TestVAE(unittest.TestCase):
    def test___intit__(self):

        self.assertTrue(True)

        # Constructs with all dimensions provided
        # activations and initializers using strings
        model = VAE(
            input_regression_dimensions=5,
            input_boolean_dimension=6,
            input_cateorical_dimension=(2, 3),
            ouput_regression_dimension=8,
            output_boolean_dimension=9,
            output_categorical_dimension=(4, 5),
            latent_dim=15,
            embedding_activations="tanh",
        )

        self.assertEqual(model.input_dimenson, 5 + 6 + 2 + 3)
        self.assertEqual(model.input_categorical_dimensions, 6)

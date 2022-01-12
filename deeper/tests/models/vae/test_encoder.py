from __future__ import annotations
import tensorflow as tf
import unittest
import numpy as np
from deeper.models.vae.encoder import VaeEncoderNet

N_ROWS = 25
DIM_X = 10
LATENT_DIM = 6
EMB_DIM = [2, 3, 4]


config = VaeEncoderNet.Config(
    latent_dim=LATENT_DIM,
    embedding_dimensions=EMB_DIM,
    embedding_activations=tf.keras.layers.ReLU(),
)


class TestVaeEncoderNet(unittest.TestCase):
    def setUp(self):
        state = np.random.RandomState(0)
        self.X = state.random((N_ROWS, DIM_X))
        self.network = VaeEncoderNet(config)

    def test_outputShape(self):
        y = self.network(self.X)

        self.assertEqual(y.logprob.shape, (N_ROWS,))
        self.assertEqual(y.prob.shape, (N_ROWS,))

        self.assertEqual(y.sample.shape, (N_ROWS, LATENT_DIM))
        self.assertEqual(y.mu.shape, (N_ROWS, LATENT_DIM))
        self.assertEqual(y.logvar.shape, (N_ROWS, LATENT_DIM))
        self.assertEqual(y.var.shape, (N_ROWS, LATENT_DIM))


if __name__ == "__main__":
    # USE CPU ONLY
    tf.config.set_visible_devices([], "GPU")
    unittest.main()

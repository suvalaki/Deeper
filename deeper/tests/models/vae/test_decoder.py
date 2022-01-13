from __future__ import annotations
import tensorflow as tf
import unittest
import numpy as np
from deeper.models.vae.base import MultipleObjectiveDimensions
from deeper.models.vae.encoder import VaeEncoderNet
from deeper.models.vae.decoder import VaeReconstructionNet
from deeper.models.vae.decoder_loss import VaeReconLossNet

N_ROWS = 25
DIM_REG = 10
DIM_BOOL = 15
DIM_ORD = (5,)
DIM_CAT = (10, 8)
DIM_X = DIM_REG + DIM_BOOL + sum(DIM_ORD) + sum(DIM_CAT)
EMB_DIM = [10, 8, 7]
LAT_DIM = 5
NCATS = 5


enc_config = VaeEncoderNet.Config(
    latent_dim=LAT_DIM,
    embedding_dimensions=EMB_DIM,
    embedding_activations=tf.keras.layers.ReLU(),
)
dec_config = VaeReconstructionNet.Config(
    output_dimensions=MultipleObjectiveDimensions(
        regression=DIM_REG,
        boolean=DIM_BOOL,
        ordinal=DIM_ORD,
        categorical=DIM_CAT,
    ),
    decoder_embedding_dimensions=EMB_DIM,
)


class TestVaeDecoderNet(unittest.TestCase):
    def setUp(self):
        state = np.random.RandomState(0)
        self.X = state.random((N_ROWS, DIM_X))
        self.network = VaeEncoderNet(enc_config)
        self.decoder = VaeReconstructionNet(dec_config)

    def test_outputShape(self):
        enc = self.network(self.X)
        y = self.decoder(enc)
        self.assertEqual(y.hidden_logits.shape, (N_ROWS, DIM_X))


if __name__ == "__main__":
    # USE CPU ONLY
    tf.config.set_visible_devices([], "GPU")
    unittest.main()

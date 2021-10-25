from __future__ import annotations
import tensorflow as tf
import unittest


# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type="CPU")
tf.config.experimental.set_visible_devices(devices=my_devices, device_type="CPU")

# To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)

import numpy as np
from deeper.models.vae.encoder import VaeEncoderNet


state = np.random.RandomState(0)
X = state.random((25, 10))
config = VaeEncoderNet.Config(
    latent_dim=6, embedding_dimensions=[2, 3, 4], embedding_activations=tf.nn.relu
)
network = VaeEncoderNet.from_config(config)
y = network(X)

from deeper.models.vae.encoder_loss import VaeLossNetLatent

lossnet = VaeLossNetLatent()

lossnet.Input.from_VaeEncoderNet(y)

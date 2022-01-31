import tensorflow as tf

import pytest
import unittest
import numpy as np
import os

from deeper.models.gan.descriminator import DescriminatorNet


DEFAULT_CONFIG = DescriminatorNet.Config(
    embedding_dimensions=[20, 15],
)


class TestDescriminatorNet(unittest.TestCase):
    def setUp(self):
        self.X = np.random.random((20, 25))
        self.y = (np.random.random((20, 1)) > 0.5).astype("float32")

    def test_outputShapes(self):
        network = DescriminatorNet(DEFAULT_CONFIG)
        preds = network(self.X)
        self.assertEqual(preds.logits.shape, (20, 1))
        self.assertEqual(preds.logprob.shape, (20, 1))
        self.assertEqual(preds.prob.shape, (20, 1))
        self.assertEqual(preds.entropy.shape, (20, 1))

        preds_with_y = network(self.X, False, self.y)
        self.assertEqual(preds_with_y.logits.shape, (20, 1))
        self.assertEqual(preds_with_y.logprob.shape, (20, 1))
        self.assertEqual(preds_with_y.prob.shape, (20, 1))
        self.assertEqual(preds_with_y.entropy.shape, (20, 1))

    def test_no_nans_for_simple_inputs(self):
        network = DescriminatorNet(DEFAULT_CONFIG)
        preds = network(self.X)
        # No Nans
        self.assertEqual(np.count_nonzero(np.isnan(preds.logits)), 0)
        self.assertEqual(np.count_nonzero(np.isnan(preds.logprob)), 0)
        self.assertEqual(np.count_nonzero(np.isnan(preds.prob)), 0)
        self.assertEqual(np.count_nonzero(np.isnan(preds.entropy)), 0)

        preds_with_y = network(self.X, False, self.y)
        self.assertEqual(np.count_nonzero(np.isnan(preds_with_y.logits)), 0)
        self.assertEqual(np.count_nonzero(np.isnan(preds_with_y.logprob)), 0)
        self.assertEqual(np.count_nonzero(np.isnan(preds_with_y.prob)), 0)
        self.assertEqual(np.count_nonzero(np.isnan(preds_with_y.entropy)), 0)

    def test_probs_between_0_and_1(self):
        network = DescriminatorNet(DEFAULT_CONFIG)
        preds = network(self.X)
        # No Nans
        self.assertEqual(tf.reduce_sum(tf.cast((preds.prob < 0.0), "float32")), 0)
        self.assertEqual(tf.reduce_sum(tf.cast((preds.prob > 1.0), "float32")), 0)
        self.assertEqual(tf.reduce_sum(tf.cast((preds.logprob > 1.0), "float32")), 0)

        preds_with_y = network(self.X, False, self.y)
        self.assertEqual(tf.reduce_sum(tf.cast((preds_with_y.prob < 0.0), "float32")), 0)
        self.assertEqual(tf.reduce_sum(tf.cast((preds_with_y.prob > 1.0), "float32")), 0)
        self.assertEqual(tf.reduce_sum(tf.cast((preds_with_y.logprob > 1.0), "float32")), 0)


if __name__ == "__main__":
    # USE CPU ONLY
    tf.config.set_visible_devices([], "GPU")
    unittest.main()

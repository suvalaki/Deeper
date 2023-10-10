from typing import Tuple, NamedTuple

import pytest
import unittest
import numpy as np
import os


import tensorflow as tf


from deeper.utils.tf.experimental.extension_type import ExtensionTypeIterableMixin


class MockNT(NamedTuple):
    var0: tf.Tensor
    var1: tf.Tensor


class MockExtension0(tf.experimental.ExtensionType, ExtensionTypeIterableMixin):
    var0: tf.Tensor


class MockExtension1(tf.experimental.ExtensionType, ExtensionTypeIterableMixin):
    var0: tf.Tensor
    var1: Tuple[tf.Tensor, tf.Tensor]
    var3: MockExtension0


class TestExtensionTypeIterableMixin(unittest.TestCase):

    def test_reducer_over_list(self):
        vals = [
            MockExtension1(
                var0=tf.constant([float(i)]),
                var1=(tf.constant([float(i)]), tf.constant([float(i)])),
                var3=MockExtension0(
                    var0=tf.constant([float(i)])
                )
            )
            for i in range(10)
        ]
        reduced = MockExtension1.reduce(vals)

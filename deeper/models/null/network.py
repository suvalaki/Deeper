from __future__ import annotations
import tensorflow as tf

from typing import Any
from pydantic import BaseModel, Field

from deeper.layers.data_splitter import SplitCovariates
from deeper.models.null.utils import NullTypeGetter
from deeper.models.generalised_autoencoder.base import LatentParser
from deeper.utils.tf.experimental.extension_type import ExtensionTypeIterableMixin

# Nullnet takes an input and replaces it with a null tensor - Can be used to skip
# Computation of an network.


class NullNet(tf.keras.layers.Layer, NullTypeGetter):
    class Config(BaseModel, NullTypeGetter):
        class Config:
            arbitrary_types_allowed = True

    class Output(tf.experimental.ExtensionType, ExtensionTypeIterableMixin, NullTypeGetter):
        identity: tf.Tensor

    def __init__(self, *args, **kwargs):
        super().__init__()

    def split_outputs(self, y) -> SplitCovariates:
        return SplitCovariates()

    def call(self, x, training=False):
        x = tf.zeros_like(tf.cast(x, dtype=self.dtype))[:, 0:0]
        return self.Output(identity=x)


class NullLatentParser(LatentParser, NullTypeGetter):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self, x: IdentityNet.Output, training=False):
        return x.identity


class NullLossNet(tf.keras.layers.Layer, NullTypeGetter):
    class InputWeight(tf.experimental.ExtensionType, ExtensionTypeIterableMixin, NullTypeGetter):
        ...

    class Input(tf.experimental.ExtensionType, ExtensionTypeIterableMixin, NullTypeGetter):
        ...

        @classmethod
        def from_output(cls, *args, **kwargs):
            return cls()

    class Output(tf.experimental.ExtensionType, ExtensionTypeIterableMixin, NullTypeGetter):
        loss: tf.Tensor = 0.0

    def __init__(self, *args, **kwargs):
        super().__init__()

    def split_outputs(self, x):
        return SplitCovariates()

    def call(self, x, training=False):
        return self.Output()


class Null(tf.keras.models.Model, NullTypeGetter):
    class Config(NullNet.Config):
        ...

    class CoolingRegime(tf.keras.layers.Layer):
        def __init__(self, *args, **kwargs):
            super().__init__(**kwargs)

        def call(self, step):
            return None

    def __init__(self, *args, **kwargs):
        super().__init__()
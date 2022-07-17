from __future__ import annotations
import tensorflow as tf

from typing import Any
from pydantic import BaseModel, Field

from deeper.layers.data_splitter import SplitCovariates
from deeper.models.identity.utils import IdentityTypeGetter
from deeper.models.generalised_autoencoder.base import LatentParser
from deeper.utils.tf.experimental.extension_type import ExtensionTypeIterableMixin


class IdentityNet(tf.keras.layers.Layer, IdentityTypeGetter):
    class Config(IdentityTypeGetter, TunableModelMixin):
        class Config:
            arbitrary_types_allowed = True

    class Output(tf.experimental.ExtensionType, ExtensionTypeIterableMixin, IdentityTypeGetter):
        identity: tf.Tensor

    def __init__(self, *args, **kwargs):
        super().__init__()

    def split_outputs(self, y) -> SplitCovariates:
        return SplitCovariates()

    def call(self, x, training=False):
        x = tf.cast(x, dtype=self.dtype)
        return self.Output(identity=x)


class IdentityLatentParser(LatentParser, IdentityTypeGetter):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self, x: IdentityNet.Output, training=False):
        return x.identity


class IdentityLossNet(tf.keras.layers.Layer, IdentityTypeGetter):
    class InputWeight(
        tf.experimental.ExtensionType, ExtensionTypeIterableMixin, IdentityTypeGetter
    ):
        ...

    class Input(tf.experimental.ExtensionType, ExtensionTypeIterableMixin, IdentityTypeGetter):
        identity: tf.Tensor

        @classmethod
        def from_output(cls, y_true, model_output, weights, *args, **kwargs):
            return cls(model_output.identity)

    class Output(tf.experimental.ExtensionType, ExtensionTypeIterableMixin, IdentityTypeGetter):
        loss: tf.Tensor = 0.0

    def __init__(self, *args, **kwargs):
        super().__init__()

    def split_outputs(self, x):
        return SplitCovariates()

    def call(self, x, training=False):
        return self.Output(tf.reduce_mean(tf.zeros_like(x.identity), axis=-1))


class Identity(tf.keras.models.Model, IdentityTypeGetter):
    class Config(IdentityNet.Config):
        ...

    class CoolingRegime(tf.keras.layers.Layer):
        def __init__(self, *args, **kwargs):
            super().__init__(**kwargs)

        def call(self, step):
            return None

    def __init__(self, *args, **kwargs):
        super().__init__()

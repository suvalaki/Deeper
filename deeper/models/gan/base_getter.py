from __future__ import annotations

import tensorflow as tf
from typing import NamedTuple, Callable, Union

from abc import ABC, abstractmethod


class BaseGanFakeOutputGetter(tf.keras.layers.Layer, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def call(
        self,
        y_pred: Union[tf.Tensor, NamedTuple, tf.experimental.ExtensionType],
        training=False,
    ) -> tf.Tensor:
        ...


class BaseGanRealOutputGetter(tf.keras.layers.Layer, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def call(
        self,
        x: Union[tf.Tensor, NamedTuple, tf.experimental.ExtensionType],
        y: Union[tf.Tensor, NamedTuple, tf.experimental.ExtensionType],
        y_pred: Union[tf.Tensor, NamedTuple, tf.experimental.ExtensionType],
        training=False,
    ) -> tf.Tensor:
        ...


class GanTypeGetter(ABC):
    @abstractmethod
    def get_generatornet_type(self):
        ...

    @abstractmethod
    def get_real_output_getter(self):
        ...

    @abstractmethod
    def get_fake_output_getter(self):
        ...

from __future__ import annotations

import tensorflow as tf
from typing import NamedTuple, Callable, Union
from abc import ABC, abstractmethod

from deeper.models.gan.base_getter import (
    BaseGanFakeOutputGetter,
    BaseGanRealOutputGetter,
)


class AdversarialAutoencoderReconstructionLossGetter(tf.keras.layers.Layer, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def call(
        self,
        lossnet_out: Union[tf.Tensor, NamedTuple, tf.experimental.ExtensionType],
        training=False,
    ) -> tf.Tensor:
        ...


class AdversarialAutoencoderTypeGetter(ABC):

    # Also prenent in GanGetterMixin
    @abstractmethod
    def get_generatornet_type(self):
        ...

    @abstractmethod
    def get_adversarialae_real_output_getter(self):
        ...

    @abstractmethod
    def get_adversarialae_fake_output_getter(self):
        ...

    @abstractmethod
    def get_adversarialae_recon_loss_getter(self):
        ...

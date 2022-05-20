from __future__ import annotations
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from typing import Union, Tuple, Sequence
from pydantic import BaseModel
from functools import singledispatchmethod as overload

from deeper.models.identity.utils import IdentityTypeGetter
from deeper.models.identity.network import IdentityNet


class Identity(tf.keras.Model, IdentityTypeGetter):
    class Config(IdentityNet.Config):
        ...

    @overload
    def __init__(self, network: IdentityNet, config, **kwargs):
        tf.keras.Model.__init__(self, **kwargs)
        self.config = config
        self.network = network

    @__init__.register
    def from_config(self, config: BaseModel, **kwargs):
        network = IdentityNet(config, **kwargs)
        self.__init__(network, config, **kwargs)

    def train_step(self, *args, **kwargs):
        return {}

    def test_step(self, *args, **kwargs):
        return {}
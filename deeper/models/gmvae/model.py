from __future__ import annotations
import tensorflow as tf
import tensorflow_addons as tfa

from typing import NamedTuple, Sequence
from pydantic import BaseModel

from .base import GmvaeModelBase
from .gmvae_marginalised_categorical import StackedGmvae, StackedGmvaeNet
from .gmvae_pure_sampling import GumbleGmvae, GumbleGmvaeNet


class InvalidModelTypeError(ValueError):
    pass


class Gmvae(GmvaeModelBase):

    VALID_OPTIONS = ["gumble", "stacked"]

    def __new__(cls, config: Gmvae.Config, **kwargs):

        return config.get_model_type()(config, **kwargs)
        raise InvalidModelTypeError(
            "Invalid Config Supplied. Use GumbleGmvae.Config or StackedGmvae.Config"
        )

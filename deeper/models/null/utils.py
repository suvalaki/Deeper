import tensorflow as tf
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
import io
from typing import Union, Tuple, Sequence
from collections import namedtuple
from deeper.models.generalised_autoencoder.base import (
    AutoencoderTypeGetterBase,
)
from deeper.models.gan.base_getter import GanTypeGetter
from deeper.models.adversarial_autoencoder.base_getter import AdversarialAutoencoderTypeGetter
from deeper.utils.tf.experimental.extension_type import ExtensionTypeIterableMixin


class NullTypeGetter(AutoencoderTypeGetterBase):
    @classmethod
    def get_cooling_regime(self):
        from deeper.models.null import Identity

        return Identity.CoolingRegime

    # Autoencoder Getters Mixin

    @classmethod
    def get_network_type(self):
        from deeper.models.null.network import NullNet

        return IdentityNet

    @classmethod
    def get_lossnet_type(self):
        from deeper.models.null.network import NullLossNet

        return IdentityLossNet

    @classmethod
    def get_model_type(self):
        from deeper.models.null.model import Null

        return Identity

    @classmethod
    def get_latent_parser_type(self):
        from deeper.models.null.network import NullLatentParser

        return IdentityLatentParser

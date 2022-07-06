import io
from collections import namedtuple
from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from deeper.models.adversarial_autoencoder.base_getter import AdversarialAutoencoderTypeGetter
from deeper.models.gan.base_getter import GanTypeGetter
from deeper.models.generalised_autoencoder.base import AutoencoderTypeGetterBase
from deeper.utils.tf.experimental.extension_type import ExtensionTypeIterableMixin
from matplotlib import pyplot as plt
from sklearn import metrics


class AutoencoderTypeGetter(AutoencoderTypeGetterBase):
    @classmethod
    def get_cooling_regime(self):
        from deeper.models.autoencoder.model import Autoencoder

        return Autoencoder.CoolingRegime

    # Autoencoder Getters Mixin

    @classmethod
    def get_network_type(self):
        from deeper.models.autoencoder.network import AutoencoderNet

        return AutoencoderNet

    @classmethod
    def get_lossnet_type(self):
        from deeper.models.autoencoder.network_loss import AutoencoderLossNet

        return AutoencoderLossNet

    @classmethod
    def get_model_type(self):
        from deeper.models.autoencoder.model import Autoencoder

        return Autoencoder

    @classmethod
    def get_latent_parser_type(self):
        from deeper.models.autoencoder.latent import AutoencoderLatentParser

        return AutoencoderLatentParser

    @classmethod
    def get_fake_output_getter(self):
        from deeper.models.autoencoder.parser import OutputParser

        return OutputParser

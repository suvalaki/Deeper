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
from deeper.utils.tf.experimental.extension_type import ExtensionTypeIterableMixin


class SplitCovariates(tf.experimental.ExtensionType, ExtensionTypeIterableMixin):
    regression: tf.Tensor
    binary: tf.Tensor
    ordinal_groups_concat: tf.Tensor
    ordinal_groups: Tuple[tf.Tensor, ...]
    categorical_groups_concat: tf.Tensor
    categorical_groups: Tuple[tf.Tensor, ...]


class VaeTypeGetter(AutoencoderTypeGetterBase, GanTypeGetter, AdversarialAutoencoderTypeGetter):
    @classmethod
    def get_cooling_regime(self):
        from deeper.models.vae.model import Vae

        return Vae.CoolingRegime

    # Autoencoder Getters Mixin

    @classmethod
    def get_network_type(self):
        from deeper.models.vae.network import VaeNet

        return VaeNet

    @classmethod
    def get_lossnet_type(self):
        from deeper.models.vae.network_loss import VaeLossNet

        return VaeLossNet

    @classmethod
    def get_model_type(self):
        from deeper.models.vae.model import Vae

        return Vae

    @classmethod
    def get_latent_parser_type(self):
        from deeper.models.vae.latent import VaeLatentParser

        return VaeLatentParser

    # Gan getters Mixin

    @classmethod
    def get_generatornet_type(self):
        from deeper.models.vae.network import VaeNet

        return VaeNet

    @classmethod
    def get_real_output_getter(self):
        from deeper.models.vae.parser import InputParser

        return InputParser

    @classmethod
    def get_fake_output_getter(self):
        from deeper.models.vae.parser import OutputParser

        return OutputParser

    # Adversarial Autoencoder getters Mixin

    @classmethod
    def get_adversarialae_real_output_getter(self):
        from deeper.models.vae.parser import LatentPriorParser

        return LatentPriorParser

    @classmethod
    def get_adversarialae_fake_output_getter(self):
        from deeper.models.vae.parser import LatentPosteriorParser

        return LatentPosteriorParser

    @classmethod
    def get_adversarialae_recon_loss_getter(self):
        from deeper.models.vae.parser import ReconstructionOnlyLossOutputParser

        return ReconstructionOnlyLossOutputParser

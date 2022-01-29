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

SplitCovariates = namedtuple(
    "SplitInputs",
    [
        "regression",
        "binary",
        "ordinal_groups_concat",
        "ordinal_groups",
        "categorical_groups_concat",
        "categorical_groups",
    ],
)


class VaeTypeGetter(AutoencoderTypeGetterBase):
    def get_network_type(self):
        from deeper.models.vae.network import VaeNet

        return VaeNet

    def get_lossnet_type(self):
        from deeper.models.vae.network_loss import VaeLossNet

        return VaeLossNet

    def get_model_type(self):
        from deeper.models.vae.model import Vae

        return Vae

    def get_latent_parser_type(self):
        from deeper.models.vae.latent import VaeLatentParser

        return VaeLatentParser

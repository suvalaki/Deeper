from __future__ import annotations
import tensorflow as tf
import numpy as np
from typing import Union, Tuple, Sequence, NamedTuple

from deeper.probability_layers.ops.normal import std_normal_kl_divergence
from deeper.layers.encoder import Encoder
from deeper.layers.binary import SigmoidEncoder

# from deeper.probability_layers.gumble_softmax import GumbleSoftmaxLayer
from deeper.probability_layers.normal import (
    RandomNormalEncoder,
    lognormal_kl,
    lognormal_pdf,
)
from deeper.utils.function_helpers.decorators import inits_args
from deeper.utils.function_helpers.collectors import get_local_tensors
from deeper.utils.scope import Scope
from deeper.models.vae.metrics import vae_categorical_dims_accuracy

from deeper.layers.data_splitter import split_inputs, unpack_dimensions
from deeper.models.vae.network import VaeNet
from deeper.models.vae.encoder_loss import VaeLossNetLatent
from deeper.models.vae.decoder_loss import VaeReconLossNet
from tensorflow.python.keras.engine import data_adapter

from types import SimpleNamespace
from deeper.utils.tf.keras.models import GenerativeModel


from tensorflow.python.keras.metrics import (
    categorical_accuracy,
)

tfk = tf.keras
Layer = tfk.layers.Layer

from deeper.models.vae.utils import SplitCovariates


class VaeLossNet(tf.keras.layers.Layer):
    def __init__(
        self,
        latent_eps=0.0,
        encoder_name="zgy",
        decoder_name="xgz",
        prefix="loss",
        **kwargs,
    ):
        super(VaeLossNet, self).__init__(**kwargs)
        self.latent_eps = latent_eps
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.prefix = prefix
        self.latent_lossnet = VaeLossNetLatent(latent_eps, name="latent_kl", **kwargs)
        self.recon_lossnet = VaeReconLossNet(decoder_name, prefix, **kwargs)

    @tf.function
    def log_pxgz(
        self,
        log_pxgz_reg,
        log_pxgz_bin,
        log_pxgz_ord,
        log_pxgz_cat,
        lambda_reg=1.0,
        lambda_bin=1.0,
        lambda_ord=1.0,
        lambda_cat=1.0,
    ):
        log_p = log_pxgz_reg + log_pxgz_bin + log_pxgz_ord + log_pxgz_cat
        self.add_metric(log_p, name=f"{self.prefix}/log_p/{self.decoder_name}")

    @tf.function
    def elbo(
        self,
        kl_z,
        ln_pxgz,
    ):
        result = ln_pxgz + kl_z
        self.add_metric(result, name=f"{self.prefix}/elbo")
        return result

    class Output(NamedTuple):
        kl_z: tf.Tensor
        l_pxgz_reg: tf.Tensor
        l_pxgz_bin: tf.Tensor
        l_pxgz_ord: tf.Tensor
        l_pxgz_cat: tf.Tensor
        scaled_l_pxgz: tf.Tensor
        scaled_elbo: tf.Tensor
        loss: tf.Tensor
        lambda_z: tf.Tensor
        lambda_reg: tf.Tensor
        lambda_bin: tf.Tensor
        lambda_ord: tf.Tensor
        lambda_cat: tf.Tensor

    @tf.function
    def loss(
        self,
        kl_z,
        log_pxgz_reg,
        log_pxgz_bin,
        log_pxgz_ord,
        log_pxgz_cat,
        lambda_z=1.0,
        lambda_reg=1.0,
        lambda_bin=1.0,
        lambda_ord=1.0,
        lambda_cat=1.0,
    ):
        scaled_kl_z = lambda_z * kl_z
        self.add_metric(scaled_kl_z, name="scaled_latent_kl")
        scaled_log_pxgz_reg = tf.cast(lambda_reg, self.dtype) * log_pxgz_reg
        self.add_metric(
            scaled_log_pxgz_reg,
            name=f"{self.prefix}/scaled/log_p_{self.decoder_name}_reg",
        )
        scaled_log_pxgz_bin = tf.cast(lambda_bin, self.dtype) * log_pxgz_bin
        self.add_metric(
            scaled_log_pxgz_bin,
            name=f"{self.prefix}/scaled/log_p_{self.decoder_name}_bin",
        )
        scaled_log_pxgz_ord = tf.cast(lambda_ord, self.dtype) * log_pxgz_ord
        self.add_metric(
            scaled_log_pxgz_ord,
            name=f"{self.prefix}/scaled/log_p_{self.decoder_name}_ord",
        )
        scaled_log_pxgz_cat = tf.cast(lambda_cat, self.dtype) * log_pxgz_cat
        self.add_metric(
            scaled_log_pxgz_cat,
            name=f"{self.prefix}/scaled/log_p_{self.decoder_name}_cat",
        )
        scaled_log_pgz = (
            scaled_log_pxgz_reg + scaled_log_pxgz_bin + scaled_log_pxgz_ord + scaled_log_pxgz_cat
        )
        self.add_metric(
            scaled_log_pgz,
            name=f"{self.prefix}/scaled/log_p{self.encoder_name}",
        )
        scaled_elbo = scaled_log_pgz + scaled_kl_z
        self.add_metric(scaled_elbo, name=f"{self.prefix}/scaled/elbo")
        scaled_loss = -scaled_elbo
        self.add_metric(scaled_loss, name=f"{self.prefix}/scaled/loss")

        # log weights
        self.add_metric(lambda_z, name=f"{self.prefix}/weight/lambda_z")
        self.add_metric(lambda_reg, name=f"{self.prefix}/weight/lambda_reg")
        self.add_metric(lambda_bin, name=f"{self.prefix}/weight/lambda_bin")
        self.add_metric(lambda_ord, name=f"{self.prefix}/weight/lambda_ord")
        self.add_metric(lambda_cat, name=f"{self.prefix}/weight/lambda_cat")

        return VaeLossNet.Output(
            kl_z,
            log_pxgz_reg,
            log_pxgz_bin,
            log_pxgz_ord,
            log_pxgz_cat,
            scaled_log_pgz,
            scaled_elbo,
            scaled_loss,
            lambda_z,
            lambda_reg,
            lambda_bin,
            lambda_ord,
            lambda_cat,
        )

    class InputWeight(NamedTuple):
        lambda_z: float = 1.0
        lambda_reg: float = 1.0
        lambda_bin: float = 1.0
        lambda_ord: float = 1.0
        lambda_cat: float = 1.0

    class Input(NamedTuple):
        latent: VaeLossNetLatent.Input
        y_true: VaeReconLossNet.InputYTrue
        y_pred: VaeReconLossNet.InputYPred
        weight: VaeLossNet.InputWeight

        @staticmethod
        @tf.function
        def from_nested_sequence(inputs) -> VaeLossNet.Input:
            return VaeLossNet.Input(
                VaeLossNetLatent.Input(*inputs[0]),
                VaeReconLossNet.InputYTrue(*inputs[1]),
                VaeReconLossNet.InputYPred(*inputs[2]),
                VaeLossNet.InputWeight(*inputs[3]),
            )

        @staticmethod
        @tf.function
        def from_output(
            y_true: SplitCovariates,
            model_output: VaeNet.VaeNetOutput,
            weights: VaeLossNet.InputWeight,
        ) -> VaeLossNet.Input:
            return VaeLossNet.Input(
                VaeLossNetLatent.Input(model_output.qz_g_x.mu, model_output.qz_g_x.logvar),
                VaeReconLossNet.InputYTrue(
                    y_true.regression,
                    y_true.binary,
                    y_true.ordinal_groups,
                    y_true.categorical_groups,
                ),
                VaeReconLossNet.InputYPred.from_VaeReconstructionNet(model_output.px_g_z),
                weights,
            )

    @tf.function
    def call(self, inputs: Input, training=False) -> Output:

        if not isinstance(inputs, VaeLossNet.Input):
            inputs = VaeLossNet.Input.from_nested_sequence(inputs)

        kl_z = self.latent_lossnet(inputs.latent, training)
        (l_pxgz_reg, l_pxgz_bin, l_pxgz_ord, l_pxgz_cat) = self.recon_lossnet(
            VaeReconLossNet.Input(inputs.y_true, inputs.y_pred), training
        )

        return self.loss(
            kl_z,
            l_pxgz_reg,
            l_pxgz_bin,
            l_pxgz_ord,
            l_pxgz_cat,
            *inputs.weight,
        )

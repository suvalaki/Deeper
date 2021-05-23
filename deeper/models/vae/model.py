import tensorflow as tf
import numpy as np
from typing import Union, Tuple, Sequence

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

from deeper.models.vae.utils import split_inputs
from deeper.models.vae.network import VaeNet, VaeLossNet
from tensorflow.python.keras.engine import data_adapter

from types import SimpleNamespace
from deeper.utils.tf.keras.models import GenerativeModel


from tensorflow.python.keras.metrics import (
    categorical_accuracy,
)

tfk = tf.keras
Layer = tfk.layers.Layer

from collections import namedtuple
from deeper.models.vae.utils import SplitCovariates


class VAE(GenerativeModel):
    @inits_args
    def __init__(
        self,
        input_regression_dimension: int,
        input_boolean_dimension: int,
        input_ordinal_dimension: Union[int, Sequence[int]],
        input_categorical_dimension: Union[int, Tuple[int]],
        output_regression_dimension: int,
        output_boolean_dimension: int,
        output_ordinal_dimension: Union[int, Sequence[int]],
        output_categorical_dimension: Union[int, Tuple[int]],
        encoder_embedding_dimensions: Tuple[int],
        decoder_embedding_dimensions: Tuple[int],
        latent_dim: int,
        embedding_activations=tf.nn.relu,
        optimizer=tf.keras.optimizers.Adam(1e-3),
        gradient_clip=None,
        **network_kwargs,
    ):
        GenerativeModel.__init__(self)
        self.cooling_distance = 0
        self.network = VaeNet(
            input_regression_dimension,
            input_boolean_dimension,
            input_ordinal_dimension,
            input_categorical_dimension,
            output_regression_dimension,
            output_boolean_dimension,
            output_ordinal_dimension,
            output_categorical_dimension,
            encoder_embedding_dimensions,
            decoder_embedding_dimensions,
            latent_dim,
            embedding_activations,
            **network_kwargs,
        )
        self.lossnet = VaeLossNet(latent_eps=1e-6)

    def increment_cooling(self):
        self.cooling_distance += 1

    @tf.function
    def sample_one(self, x, training=False):
        return self.network(x, training)

    @tf.function
    def loss_fn_weighted(
        self, weights, y_true, y_pred: VaeNet.VaeNetOutput, training=False
    ) -> VaeLossNet.output:
        y_true = tf.cast(y_true, dtype=self.dtype)
        y_split = self.network.split_outputs(y_true)

        # This can be none?
        # self.lossnet.categorical_accuracy_grouped(
        #    y_split.categorical_groups, y_pred.x_recon_cat_groups
        # )

        loss = self.lossnet.Output(
            *[
                tf.reduce_mean(x)
                for x in self.lossnet(
                    self.lossnet.Input.from_vaenet_outputs(
                        y_split,
                        y_pred,
                        self.lossnet.InputWeight(*weights),
                    ),
                    training,
                )
            ]
        )

        return loss

    @tf.function
    def loss_fn(
        self, y_true, y_pred: VaeNet.VaeNetOutput, training=False
    ) -> VaeLossNet.output:
        y_true = tf.cast(y_true, dtype=self.dtype)
        y_split = self.network.split_outputs(y_true)

        # This can be none?
        # self.lossnet.categorical_accuracy_grouped(
        #    y_split.categorical_groups, y_pred.x_recon_cat_groups
        # )

        loss = self.lossnet.Output(
            *[
                tf.reduce_mean(x)
                for x in self.lossnet(
                    self.lossnet.Input.from_vaenet_outputs(
                        y_split,
                        y_pred,
                        self.lossnet.InputWeight(1.0, 1.0, 1.0, 1.0, 1.0),
                    ),
                    training,
                )
            ]
        )

        return loss

    @tf.function
    def split_inputs(
        self, x, reg_dim: int, bool_dim: int, cat_dim_tup: Tuple[int]
    ):
        x_reg, x_bin, x_ord, x_ord_g, x_cat, x_cat_g = split_inputs(
            x, reg_dim, bool_dim, (0,), cat_dim_tup
        )
        return x_reg, x_bin, x_cat, x_cat_g

    @tf.function
    def call(self, x, training=False):
        return self.sample_one(x, training)

    @tf.function
    def latent_sample(self, inputs, y, training=False, samples=1):
        output = self.monte_carlo_estimate(
            samples, inputs, y, training=training
        )
        latent = outputs["px_g_z__sample"]
        return latent

    @tf.function
    def train_step(
        self,
        data,
        samples=1,
        tenorboard=False,
        batch=False,
        beta_reg=1.0,
        beta_bin=1.0,
        beta_cat=1.0,
        beta_z=1.0,
    ):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        if tenorboard:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = "logs/gradient_tape/train"

            writer = tf.summary.create_file_writer(train_log_dir)

        with tf.GradientTape() as tape:
            y_pred = self(x, True)
            losses = self.loss_fn(y, y_pred, True)
            loss = tf.reduce_mean(losses.loss)

        gradients = tape.gradient(loss, self.trainable_variables)

        # Clipping
        gradients = [
            None
            if gradient is None
            else tf.clip_by_value(
                gradient, -self.gradient_clip, self.gradient_clip
            )
            if self.gradient_clip is not None
            else gradient
            # else tf.clip_by_norm(
            #    gradient, self.gradient_clip
            # )
            for gradient in gradients
        ]

        if tenorboard:
            with writer.as_default():
                for gradient, variable in zip(
                    gradients, self.trainable_variables
                ):
                    steps = steps + 1
                    tf.summary.experimental.set_step(steps)
                    stp = tf.summary.experimental.get_step()
                    tf.summary.histogram(
                        "gradients/" + variable.name,
                        tf.nn.l2_normalize(gradient),
                        step=stp,
                    )
                    tf.summary.histogram(
                        "variables/" + variable.name,
                        tf.nn.l2_normalize(variable),
                        step=stp,
                    )
                writer.flush()

        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )
        # self.compiled_metrics.update_state(y, y_pred)
        # for m in self.intrinsic_metrics:
        #   m.update_state(y, y_pred)
        metval = self.metric_results
        return {
            "loss": loss,
            "latent_kl": metval["latent_kl"],
            # "log_px_reg": logpx_reg,
            # "log_px_bin": logpx_bin,
            # "log_px_cat": logpx_cat,
            **{m.name: m.result() for m in self.intrinsic_metrics},
        }
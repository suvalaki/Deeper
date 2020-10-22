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
from deeper.utils.sampling import mc_stack_mean_dict
from deeper.utils.function_helpers.decorators import inits_args
from deeper.utils.function_helpers.collectors import get_local_tensors
from deeper.utils.scope import Scope
from deeper.models.vae.metrics import VaeCategoricalAvgAccuracy

from deeper.models.vae.utils import split_inputs
from deeper.models.vae.network import VaeNet

from tensorflow.python.keras.engine import data_adapter

from types import SimpleNamespace

from deeper.utils.tf.keras.models import Model

tfk = tf.keras
Layer = tfk.layers.Layer


class VAE(Model):
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
        Model.__init__(self)
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

    def increment_cooling(self):
        self.cooling_distance += 1

    @tf.function
    def sample_one(self, x, y, training=False):

        y = tf.cast(y, dtype=self.dtype)
        (
            y_reg,
            y_bin,
            y_ord_groups_concat,
            y_groups,
            y_cat_groups_concat,
            y_cat_groups,
        ) = self.network.split_outputs(
            y,
        )

        result = self.network.call_dict(x, training)

        (
            x_recon_reg,
            x_recon_bin_logit,
            x_recon_ord_groups_logit_concat,
            x_recon_ord_groups_logit,
            x_recon_cat_logit_groups_concat,
            x_recon_cat_logit_groups,
        ) = self.network.split_outputs(
            result["x_recon"],
        )

        x_recon_cat_groups = [
            tf.nn.softmax(x) for x in x_recon_cat_logit_groups
        ]

        # Calculate reconstruction epsilon assuming independence between distributions
        log_px_recon_regression = lognormal_pdf(x_recon_reg, y_reg, 1.0, 1e-6)
        x_bin_xent = (
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y_bin,
                logits=x_recon_bin_logit,
                name="recon_sigmoid_crossent",
            )
            if self.output_boolean_dimension > 0
            else tf.zeros(tf.shape(x_recon_bin_logit))
        )

        x_cat_xents = (
            tf.zeros(tf.shape(x_recon_cat_logit_groups_concat))
            if self.network.output_categorical_dimension == 0
            else tf.nn.softmax_cross_entropy_with_logits(
                y_cat_groups[0], x_recon_cat_logit_groups[0]
            )
            if len(self.network.input_categorical_dimension) == 1
            else tf.reduce_sum(
                tf.concat(
                    [
                        tf.nn.softmax_cross_entropy_with_logits(y, x, -1)
                        for x, y in zip(y_cat_groups, x_recon_cat_logit_groups)
                    ],
                    -1,
                ),
                -1,
            )
        )
        recon = px_g_z__logprob = (
            log_px_recon_regression
            - tf.reduce_sum(x_bin_xent, -1)
            - tf.reduce_sum(x_cat_xents, -1)
        )
        px_g_z__prob = tf.exp(px_g_z__logprob)
        z_entropy = std_normal_kl_divergence(
            result["qz_g_x__mu"], result["qz_g_x__logvar"]
        )
        elbo = px_g_z__logprob + z_entropy
        output = {
            **result,
            **{
                "y_reg": y_reg,
                "y_bin": y_bin,
                "y_cat_groups_concat": y_cat_groups_concat,
                "log_px_recon_regression": log_px_recon_regression,
                "x_bin_xent": x_bin_xent,
                "x_cat_xents": x_cat_xents,
                "px_g_z__logprob": px_g_z__logprob,
                "px_g_z__prob": px_g_z__prob,
                "recon": recon,
                "z_entropy": z_entropy,
                "elbo": elbo,
            },
        }

        return output

    @tf.function
    def split_inputs(
        self, x, reg_dim: int, bool_dim: int, cat_dim_tup: Tuple[int]
    ):
        x_reg, x_bin, x_ord, x_ord_g, x_cat, x_cat_g = split_inputs(
            x, reg_dim, bool_dim, (0,), cat_dim_tup
        )
        return x_reg, x_bin, x_cat, x_cat_g

    @tf.function
    def sample(self, samples, x, y, training=False):
        # with tf.device("/gpu:0"):
        result = [self.sample_one(x, y, training) for j in range(samples)]
        return result

    @tf.function
    def monte_carlo_estimate(self, samples, x, y, training=False):
        return mc_stack_mean_dict(self.sample(samples, x, y, training))

    @tf.function
    def call(self, x, training=False):
        return self.network(x, training)

    @tf.function
    def latent_sample(self, inputs, y, training=False, samples=1):
        output = self.monte_carlo_estimate(
            samples, inputs, y, training=training
        )
        latent = outputs["px_g_z__sample"]
        return latent

    @tf.function
    def entropy_fn(self, inputs, y, training=False, samples=1):
        output = self.monte_carlo_estimate(
            samples, inputs, y, training=training
        )
        return (
            output["recon"],
            output["log_px_recon_regression"],
            output["x_bin_xent"],
            output["x_cat_xents"],
            output["z_entropy"],
        )

    @tf.function
    def elbo(
        self,
        inputs,
        y,
        training=False,
        samples=1,
        beta_reg=1.0,
        beta_bin=1.0,
        beta_cat=1.0,
        beta_z=1.0,
    ):
        recon, logpx_reg, bin_xent, cat_xent, z_entropy = self.entropy_fn(
            inputs, y, training, samples
        )
        recon = px_g_z__logprob = (
            beta_reg * logpx_reg
            - beta_bin * tf.reduce_sum(bin_xent, -1)
            - beta_cat * tf.reduce_sum(cat_xent, -1)
        )
        return recon + beta_z * z_entropy

    @tf.function
    def losses_fns(
        self,
        output,
        beta_reg=1.0,
        beta_bin=1.0,
        beta_cat=1.0,
        beta_z=1.0,
    ):
        recon, logpx_reg, bin_xent, cat_xent, z_entropy = (
            output["recon"],
            output["log_px_recon_regression"],
            output["x_bin_xent"],
            output["x_cat_xents"],
            output["z_entropy"],
        )
        logpx_bin = tf.reduce_sum(bin_xent, -1)
        logpx_cat = tf.reduce_sum(cat_xent, -1)
        recon = px_g_z__logprob = (
            beta_reg * logpx_reg - beta_bin * logpx_bin - beta_cat * logpx_cat
        )
        elbo = recon + beta_z * z_entropy
        loss = -elbo
        return loss, recon, logpx_reg, logpx_bin, logpx_cat, z_entropy

    @tf.function
    def loss_fn(
        self,
        inputs,
        y,
        training=False,
        samples=1,
        beta_reg=1.0,
        beta_bin=1.0,
        beta_cat=1.0,
        beta_z=1.0,
    ):
        return -self.elbo(inputs, y, training, samples, beta_z)

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
            y_pred = self.monte_carlo_estimate(samples, x, y, True)
            loss, recon, logpx_reg, logpx_bin, logpx_cat, z_entropy = [
                tf.reduce_mean(output)
                for output in self.losses_fns(
                    y_pred, beta_reg, beta_bin, beta_cat, beta_z
                )
            ]

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
        #    m.update_state(y, y_pred)
        return {
            "loss": loss,
            # "log_px": recon,
            # "log_px_reg": logpx_reg,
            # "log_px_bin": logpx_bin,
            # "log_px_cat": logpx_cat,
            "z_kl_entropy": z_entropy,
            **{m.name: m.result() for m in self.intrinsic_metrics},
        }
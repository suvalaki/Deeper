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
from deeper.models.vae.metrics import vae_categorical_dims_accuracy

from deeper.models.vae.utils import split_inputs
from deeper.models.vae.network import VaeNet, VaeLossNet
from tensorflow.python.keras.engine import data_adapter

from types import SimpleNamespace

from deeper.utils.tf.keras.models import Model


from tensorflow.python.keras.metrics import (
    categorical_accuracy,
)

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
        self.lossnet = VaeLossNet(latent_eps=1e-6)

    def increment_cooling(self):
        self.cooling_distance += 1

    @tf.function
    def sample_one(self, x, y, training=False):

        y = tf.cast(y, dtype=self.dtype)
        (
            y_reg,
            y_bin,
            y_ord_groups_concat,
            y_ord_groups,
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
        self.lossnet.categorical_accuracy_grouped(
            y_cat_groups, x_recon_cat_groups
        )

        loss = tf.reduce_mean(
            self.lossnet(
                (
                    (result["qz_g_x__mu"], result["qz_g_x__logvar"]),
                    (y_reg, y_bin, y_ord_groups, y_cat_groups),
                    (
                        x_recon_reg,
                        x_recon_bin_logit,
                        x_recon_ord_groups_logit,
                        x_recon_cat_logit_groups,
                    ),
                    (1.0, 1.0, 1.0, 1.0, 1.0),
                ),
                training,
            )
        )
        # self.add_loss(loss)

        output = {
            **result,
            **{
                "loss": tf.reduce_mean(loss),
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
            y_pred = self.sample_one(x, y, True)
            """
            y_pred = self.monte_carlo_estimate(samples, x, y, True)
            loss, recon, logpx_reg, logpx_bin, logpx_cat, z_entropy = [
                tf.reduce_mean(output)
                for output in self.losses_fns(
                    y_pred, beta_reg, beta_bin, beta_cat, beta_z
                )
            ]
            """
            loss = y_pred["loss"]

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
            **{m.name: m.result() for m in self.intrinsic_metrics},
        }
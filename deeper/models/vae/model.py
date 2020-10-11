import tensorflow as tf
import numpy as np
from typing import Union, Tuple

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

tfk = tf.keras
Model = tfk.Model
Layer = tfk.layers.Layer


class VAE(Model, Scope):
    @inits_args
    def __init__(
        self,
        input_regression_dimension: int,
        input_boolean_dimension: int,
        input_categorical_dimension: Union[int, Tuple[int]],
        output_regression_dimension: int,
        output_boolean_dimension: int,
        output_categorical_dimension: Union[int, Tuple[int]],
        encoder_embedding_dimensions: Tuple[int],
        decoder_embedding_dimensions: Tuple[int],
        latent_dim,
        embedding_activations=tf.nn.tanh,
        kind="binary",
        var_scope="variational_autoencoder",
        bn_before=False,
        bn_after=False,
        latent_epsilon=0.0,
        reconstruction_epsilon=0.0,
        enc_mu_embedding_kernel_initializer="glorot_uniform",
        enc_mu_embedding_bias_initializer="zeros",
        enc_mu_latent_kernel_initialiazer="glorot_uniform",
        enc_mu_latent_bias_initializer="zeros",
        enc_var_embedding_kernel_initializer="glorot_uniform",
        enc_var_embedding_bias_initializer="zeros",
        enc_var_latent_kernel_initialiazer="glorot_uniform",
        enc_var_latent_bias_initializer="zeros",
        recon_embedding_kernel_initializer="glorot_uniform",
        recon_embedding_bias_initializer="zeros",
        recon_latent_kernel_initialiazer="glorot_uniform",
        recon_latent_bias_initializer="zeros",
        connected_weights=True,
        latent_mu_embedding_dropout=0.0,
        latent_var_embedding_dropout=0.0,
        recon_dropouut=0.0,
        latent_fixed_var=None,
        optimizer=tf.keras.optimizers.Adam(1e-3),
        gradient_clip=None,
    ):
        Model.__init__(self)
        Scope.__init__(self, var_scope)

        self.cooling_distance = 0

        # Input Dimension Calculation
        self.input_categorical_dimension = (
            input_categorical_dimension
            if type(input_categorical_dimension) == tuple
            else (input_categorical_dimension,)
        )
        self.input_cat_dim = (
            input_categorical_dimension
            if type(input_categorical_dimension) == int
            else sum(input_categorical_dimension)
        )
        self.input_dim = (
            input_regression_dimension
            + input_boolean_dimension
            + self.input_cat_dim
        )
        self.output_categorical_dimension = (
            output_categorical_dimension
            if type(output_categorical_dimension) == tuple
            else (output_categorical_dimension,)
        )
        self.output_cat_dim = (
            output_categorical_dimension
            if type(output_categorical_dimension) == int
            else sum(output_categorical_dimension)
        )
        self.output_dim = (
            output_regression_dimension
            + output_boolean_dimension
            + self.output_cat_dim
        )

        # Encoder
        self.graph_qz_g_x = RandomNormalEncoder(
            latent_dimension=self.latent_dim,
            embedding_dimensions=self.encoder_embedding_dimensions,
            var_scope=self.v_name("graph_qz_g_x"),
            bn_before=self.bn_before,
            bn_after=self.bn_after,
            epsilon=self.latent_epsilon,
            embedding_mu_kernel_initializer=enc_mu_embedding_kernel_initializer,
            embedding_mu_bias_initializer=enc_mu_embedding_bias_initializer,
            latent_mu_kernel_initialiazer=enc_mu_latent_kernel_initialiazer,
            latent_mu_bias_initializer=enc_mu_latent_bias_initializer,
            embedding_var_kernel_initializer=enc_var_embedding_kernel_initializer,
            embedding_var_bias_initializer=enc_var_embedding_bias_initializer,
            latent_var_kernel_initialiazer=enc_var_latent_kernel_initialiazer,
            latent_var_bias_initializer=enc_var_latent_bias_initializer,
            connected_weights=connected_weights,
            embedding_mu_dropout=latent_mu_embedding_dropout,
            embedding_var_dropout=latent_var_embedding_dropout,
            fixed_var=latent_fixed_var,
        )

        # Decoder
        self.graph_px_g_z = Encoder(
            self.output_dim,
            self.decoder_embedding_dimensions,
            activation=self.embedding_activations,
            var_scope=self.v_name("graph_px_g_z"),
            bn_before=self.bn_before,
            bn_after=self.bn_after,
            embedding_kernel_initializer=recon_embedding_kernel_initializer,
            embedding_bias_initializer=recon_embedding_bias_initializer,
            latent_kernel_initialiazer=recon_latent_kernel_initialiazer,
            latent_bias_initializer=recon_latent_bias_initializer,
            embedding_dropout=recon_dropouut,
        )

    def increment_cooling(self):
        self.cooling_distance += 1

    @tf.function
    def predict_one(self, x, training=False):

        x = tf.cast(x, dtype=self.dtype)
        (
            x_regression,
            x_bin,
            x_cat_groups,
            x_cat_groups_concat,
        ) = self.split_inputs(
            x,
            self.input_regression_dimension,
            self.input_boolean_dimension,
            self.input_categorical_dimension,
        )

        # Encoder
        (
            qz_g_x__sample,
            qz_g_x__logprob,
            qz_g_x__prob,
            qz_g_x__mu,
            qz_g_x__logvar,
            qz_g_x__var,
        ) = self.graph_qz_g_x.call(x, training)

        # `Decoder
        out_hidden = self.graph_px_g_z.call(qz_g_x__sample, training)

        (
            x_recon_regression,
            x_recon_bin_logit,
            x_recon_cat_groups_logit,
            x_recon_cat_groups_logit_concat,
        ) = self.split_inputs(
            out_hidden,
            self.output_regression_dimension,
            self.output_boolean_dimension,
            self.output_categorical_dimension,
        )

        x_recon_bin = tf.nn.sigmoid(x_recon_bin_logit)
        x_recon_cat_groups = [
            tf.nn.softmax(x) for x in x_recon_cat_groups_logit
        ]
        x_recon_cat_groups_concat = (
            tf.nn.softmax(x_recon_cat_groups[0])
            if len(x_recon_cat_groups_logit) <= 1
            else tf.concat(
                [tf.nn.softmax(z) for z in x_recon_cat_groups_logit], -1
            )
        )

        result = {
            # Input variables
            "x_regression": x_regression,
            "x_bin": x_bin,
            "x_cat_groups_concat": x_cat_groups_concat,
            # Encoder Variables
            "qz_g_x__sample": qz_g_x__sample,
            "qz_g_x__logprob": qz_g_x__logprob,
            "qz_g_x__prob": qz_g_x__prob,
            "qz_g_x__mu": qz_g_x__mu,
            "qz_g_x__logvar": qz_g_x__logvar,
            "qz_g_x__var": qz_g_x__var,
            # DecoderVariables
            "x_recon": out_hidden,
            "x_recon_regression": x_recon_regression,
            "x_recon_bin_logit": x_recon_bin_logit,
            "x_recon_bin": x_recon_bin,
            "x_recon_cat_groups_logit_concat": x_recon_cat_groups_logit_concat,
            "x_recon_cat_groups_concat": x_recon_cat_groups_concat,
        }

        return result

    @tf.function
    def sample_one(self, x, y, training=False):

        y = tf.cast(y, dtype=self.dtype)
        y_reg, y_bin, y_cat_groups, y_cat_groups_concat = self.split_inputs(
            y,
            self.output_regression_dimension,
            self.output_boolean_dimension,
            self.output_categorical_dimension,
        )

        result = self.predict_one(x, training)

        (
            x_recon_reg,
            x_recon_bin_logit,
            x_recon_cat_logit_groups,
            x_recon_cat_logit_groups_concat,
        ) = self.split_inputs(
            result["x_recon"],
            self.output_regression_dimension,
            self.output_boolean_dimension,
            self.output_categorical_dimension,
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
            if self.output_cat_dim == 0
            else tf.nn.softmax_cross_entropy_with_logits(
                y_cat_groups[0], x_recon_cat_logit_groups[0]
            )
            if len(self.input_categorical_dimension) == 1
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
        x_regression = (
            x[:, :reg_dim] if reg_dim > 0 else tf.zeros((tf.shape(x)[0], 0))
        )
        x_bin_logit = x[
            :,
            reg_dim : (reg_dim + bool_dim),
        ]

        # categorical dimensions need to be further broken up according to the size
        # of the input groups
        cat_dim = sum(cat_dim_tup)
        x_cat_softinv = x[:, -cat_dim:] if cat_dim > 0 else x[:, 0:0]
        x_cat_softinv_groups = [
            x_cat_softinv[
                :,
                sum(cat_dim_tup[:i]) : sum(cat_dim_tup[i : i + 1]),
            ]
            for i in range(len(cat_dim_tup))
        ]
        return x_regression, x_bin_logit, x_cat_softinv_groups, x_cat_softinv

    @tf.function
    def sample(self, samples, x, y, training=False):
        # with tf.device("/gpu:0"):
        result = [self.sample_one(x, y, training) for j in range(samples)]
        return result

    @tf.function
    def monte_carlo_estimate(self, samples, x, y, training=False):
        return mc_stack_mean_dict(self.sample(samples, x, y, training))

    @tf.function
    def call(self, x, y, training=False, samples=1):
        output = self.monte_carlo_estimate(samples, x, y, training)
        return output

    @tf.function
    def latent_sample(self, inputs, y, training=False, samples=1):
        outputs = self.call(inputs, y, training=training, samples=samples)
        latent = outputs["px_g_z__sample"]
        return latent

    @tf.function
    def entropy_fn(self, inputs, y, training=False, samples=1):
        output = self.call(inputs, y, training=training, samples=samples)
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
        beta_z=1.0,
    ):
        recon, logpx_reg, bin_xent, cat_xent, z_entropy = self.entropy_fn(
            inputs, y, training, samples
        )
        return recon + beta_z * z_entropy

    @tf.function
    def loss_fn(self, inputs, y, training=False, samples=1, beta_z=1.0):
        return -self.elbo(inputs, y, training, samples, beta_z)

    @tf.function
    def train_step(
        self,
        x,
        y,
        samples=1,
        tenorboard=False,
        batch=False,
        beta_z=1.0,
    ):

        if tenorboard:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = "logs/gradient_tape/train"

            writer = tf.summary.create_file_writer(train_log_dir)

        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(self.loss_fn(x, y, True, samples, beta_z), 0)

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

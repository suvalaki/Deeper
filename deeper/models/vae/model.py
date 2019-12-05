import tensorflow as tf
import numpy as np

from deeper.probability_layers.ops.normal import std_normal_kl_divergence
from deeper.layers.binary import SigmoidEncoder
from deeper.probability_layers.gumble_softmax import GumbleSoftmaxLayer
from deeper.probability_layers.normal import RandomNormalEncoder, lognormal_kl
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
        input_dimension,
        embedding_dimensions,
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

        # Encoder
        self.graph_qz_g_x = RandomNormalEncoder(
            latent_dimension=self.latent_dim,
            embedding_dimensions=self.embedding_dimensions,
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
        if self.kind == "binary":
            self.graph_px_g_z = SigmoidEncoder(
                latent_dimension=self.input_dimension,
                embedding_dimensions=self.embedding_dimensions[::-1],
                var_scope=self.v_name("graph_px_g_z"),
                bn_before=self.bn_before,
                bn_after=self.bn_after,
                epsilon=self.reconstruction_epsilon,
                embedding_kernel_initializer=recon_embedding_kernel_initializer,
                embedding_bias_initializer=recon_embedding_bias_initializer,
                latent_kernel_initialiazer=recon_latent_kernel_initialiazer,
                latent_bias_initializer=recon_latent_bias_initializer,
                embedding_dropout=recon_dropouut,
            )
        else:
            self.graph_px_g_z = RandomNormalEncoder(
                self.input_dimension,
                self.embedding_dimensions[::-1],
                var_scope=self.v_name("graph_px_g_z"),
                bn_before=self.bn_before,
                bn_after=self.bn_after,
                embedding_mu_dropout=recon_dropouut,
                embedding_var_dropout=recon_dropouut,
                fixed_var=1.0,
                epsilon=self.reconstruction_epsilon,
                embedding_mu_kernel_initializer=recon_embedding_kernel_initializer,
                embedding_mu_bias_initializer=recon_embedding_bias_initializer,
                latent_mu_kernel_initialiazer=recon_latent_kernel_initialiazer,
                latent_mu_bias_initializer=recon_latent_bias_initializer,
            )

    def increment_cooling(self):
        self.cooling_distance += 1

    @tf.function
    def sample_one(self, x, training=False):
        x = tf.cast(x, dtype=self.dtype)
        (
            qz_g_x__sample,
            qz_g_x__logprob,
            qz_g_x__prob,
            qz_g_x__mu,
            qz_g_x__logvar,
            qz_g_x__var,
        ) = self.graph_qz_g_x.call(x, training)

        (
            px_g_z__sample,
            px_g_z__logprob,
            px_g_z__prob,
        ) = self.graph_px_g_z.call(qz_g_x__sample, training, x)[0:3]

        recon = px_g_z__logprob
        z_entropy = std_normal_kl_divergence(qz_g_x__mu, qz_g_x__logvar)
        elbo = recon + z_entropy

        output = {
            "qz_g_x__sample": qz_g_x__sample,
            "qz_g_x__logprob": qz_g_x__logprob,
            "qz_g_x__prob": qz_g_x__prob,
            "qz_g_x__mu": qz_g_x__mu,
            "qz_g_x__logvar": qz_g_x__logvar,
            "qz_g_x__var": qz_g_x__var,
            "px_g_z__sample": px_g_z__sample,
            "px_g_z__logprob": px_g_z__logprob,
            "px_g_z__prob": px_g_z__prob,
            "recon": recon,
            "z_entropy": z_entropy,
            "elbo": elbo,
        }

        return output

    @tf.function
    def sample(self, samples, x, training=False):
        # with tf.device("/gpu:0"):
        result = [self.sample_one(x, training) for j in range(samples)]
        return result

    @tf.function
    def monte_carlo_estimate(self, samples, x, training=False):
        return mc_stack_mean_dict(self.sample(samples, x, training))

    @tf.function
    def call(self, x, training=False, samples=1):
        output = self.monte_carlo_estimate(samples, x, training)
        return output

    @tf.function
    def latent_sample(self, inputs, training=False, samples=1):
        outputs = self.call(inputs, training=training, samples=samples)
        latent = outputs["px_g_z__sample"]
        return latent

    @tf.function
    def entropy_fn(self, inputs, training=False, samples=1):
        output = self.call(inputs, training=training, samples=samples)
        return output["recon"], output["z_entropy"]

    @tf.function
    def elbo(
        self, inputs, training=False, samples=1, beta_z=1.0,
    ):
        recon, z_entropy = self.entropy_fn(inputs, training, samples)
        return recon + beta_z * z_entropy

    @tf.function
    def loss_fn(self, inputs, training=False, samples=1, beta_z=1.0):
        return -self.elbo(inputs, training, samples, beta_z)

    @tf.function
    def train_step(
        self, x, samples=1, tenorboard=False, batch=False, beta_z=1.0,
    ):

        if tenorboard:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = "logs/gradient_tape/train"

            writer = tf.summary.create_file_writer(train_log_dir)

        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(self.loss_fn(x, True, samples, beta_z), 0)

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

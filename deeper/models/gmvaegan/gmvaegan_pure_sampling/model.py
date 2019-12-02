import tensorflow as tf
from tensorflow.python.eager import context
import numpy as np
import datetime

from typing import Optional, Dict

from deeper.models.gmvae.gmvae_pure_sampling.model import Gmvae
from deeper.probability_layers.gumble_softmax import GumbleSoftmaxLayer
from deeper.probability_layers.normal import RandomNormalEncoder
from deeper.layers.binary import SigmoidEncoder
from deeper.layers.categorical import CategoricalEncoder
from deeper.utils.scope import Scope
from deeper.utils.sampling import mc_stack_mean_dict
from deeper.utils.function_helpers.decorators import inits_args

tfk = tf.keras

Model = tfk.Model
Tensor = tf.Tensor


class GmvaeGan(Model, Scope):
    @inits_args
    def __init__(
        self,
        descriminator_dimensions,
        components,
        input_dimension,
        embedding_dimensions,
        latent_dimensions,
        embedding_activations=tf.nn.relu,
        mixture_embedding_activations=None,
        mixture_embedding_dimensions=None,
        mixture_latent_dimensions=None,
        bn_before=False,
        bn_after=False,
        categorical_epsilon=0.0,
        latent_epsilon=0.0,
        latent_prior_epsilon=0.0,
        reconstruction_epsilon=0.0,
        kind="binary",
        learning_rate=0.01,
        gradient_clip=None,
        var_scope="gmvaegan",
        descr_embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        descr_embedding_bias_initializer=tf.initializers.zeros(),
        descr_latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        descr_latent_bias_initializer=tf.initializers.zeros(),
        cat_embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        cat_embedding_bias_initializer=tf.initializers.zeros(),
        cat_latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        cat_latent_bias_initializer=None,
        latent_mu_embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        latent_mu_embedding_bias_initializer=tf.initializers.zeros(),
        latent_mu_latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        latent_mu_latent_bias_initializer=tf.initializers.zeros(),
        latent_var_embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        latent_var_embedding_bias_initializer=tf.initializers.zeros(),
        latent_var_latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        latent_var_latent_bias_initializer=tf.initializers.constant(1.0),
        posterior_mu_embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        posterior_mu_embedding_bias_initializer=tf.initializers.zeros(),
        posterior_mu_latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        posterior_mu_latent_bias_initializer=tf.initializers.zeros(),
        posterior_var_embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        posterior_var_embedding_bias_initializer=tf.initializers.zeros(),
        posterior_var_latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        posterior_var_latent_bias_initializer=tf.initializers.constant(1.0),
        recon_embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        recon_embedding_bias_initializer=tf.initializers.zeros(),
        recon_latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        recon_latent_bias_initializer=tf.initializers.zeros(),
        z_kl_lambda=1.0,
        c_kl_lambda=1.0,
        vae_optimizer=tf.keras.optimizers.SGD(0.001),
        gan_optimizer=tf.keras.optimizers.SGD(0.001),
        dec_optimizer=tf.keras.optimizers.SGD(0.001),
        connected_weights=True,
        categorical_latent_embedding_dropout=0.0,
        mixture_latent_mu_embedding_dropout=0.0,
        mixture_latent_var_embedding_dropout=0.0,
        mixture_posterior_mu_dropout=0.0,
        mixture_posterior_var_dropout=0.0,
        recon_dropouut=0.0,
        latent_fixed_var=None,
    ):

        self.mem_dim = (
            mixture_embedding_dimensions
            if mixture_embedding_dimensions is not None
            else self.embedding_dimensions
        )
        self.mem_act = (
            mixture_embedding_activations
            if mixture_embedding_activations is not None
            else self.embedding_activations
        )
        self.mem_lat = (
            mixture_latent_dimensions
            if mixture_latent_dimensions is not None
            else self.latent_dimensions
        )

        self.bn_before = bn_before
        self.bn_after = bn_after

        self.cat_eps = categorical_epsilon
        self.lat_eps = latent_epsilon
        self.rec_eps = reconstruction_epsilon

        self.kind = kind
        self.gradient_clip = gradient_clip
        self.learning_rate = learning_rate
        self.cooling_distance = 0

        Model.__init__(self)
        Scope.__init__(self, var_scope)

        self.gmvae = Gmvae(
            components=components,
            input_dimension=input_dimension,
            embedding_dimensions=embedding_dimensions,
            latent_dimensions=latent_dimensions,
            embedding_activations=embedding_activations,
            mixture_embedding_activations=mixture_embedding_activations,
            mixture_embedding_dimensions=mixture_embedding_dimensions,
            bn_before=bn_before,
            bn_after=bn_after,
            categorical_epsilon=categorical_epsilon,
            latent_epsilon=latent_epsilon,
            latent_prior_epsilon=latent_prior_epsilon,
            reconstruction_epsilon=reconstruction_epsilon,
            kind=kind,
            learning_rate=learning_rate,
            gradient_clip=gradient_clip,
            var_scope=self.v_name("gmvae"),
            cat_embedding_kernel_initializer=cat_embedding_kernel_initializer,
            cat_embedding_bias_initializer=cat_embedding_bias_initializer,
            cat_latent_kernel_initialiazer=cat_latent_kernel_initialiazer,
            cat_latent_bias_initializer=cat_latent_bias_initializer,
            latent_mu_embedding_kernel_initializer=latent_mu_embedding_kernel_initializer,
            latent_mu_embedding_bias_initializer=latent_mu_embedding_bias_initializer,
            latent_mu_latent_kernel_initialiazer=latent_mu_latent_kernel_initialiazer,
            latent_mu_latent_bias_initializer=latent_mu_latent_bias_initializer,
            latent_var_embedding_kernel_initializer=latent_var_embedding_kernel_initializer,
            latent_var_embedding_bias_initializer=latent_var_embedding_bias_initializer,
            latent_var_latent_kernel_initialiazer=latent_var_latent_kernel_initialiazer,
            latent_var_latent_bias_initializer=latent_var_latent_bias_initializer,
            posterior_mu_embedding_kernel_initializer=posterior_mu_embedding_kernel_initializer,
            posterior_mu_embedding_bias_initializer=posterior_mu_embedding_bias_initializer,
            posterior_mu_latent_kernel_initialiazer=posterior_mu_latent_kernel_initialiazer,
            posterior_mu_latent_bias_initializer=posterior_mu_latent_bias_initializer,
            posterior_var_embedding_kernel_initializer=posterior_var_embedding_kernel_initializer,
            posterior_var_embedding_bias_initializer=posterior_var_embedding_bias_initializer,
            posterior_var_latent_kernel_initialiazer=posterior_var_latent_kernel_initialiazer,
            posterior_var_latent_bias_initializer=posterior_var_latent_bias_initializer,
            recon_embedding_kernel_initializer=recon_embedding_kernel_initializer,
            recon_embedding_bias_initializer=recon_embedding_bias_initializer,
            recon_latent_kernel_initialiazer=recon_latent_kernel_initialiazer,
            recon_latent_bias_initializer=recon_latent_bias_initializer,
            z_kl_lambda=z_kl_lambda,
            c_kl_lambda=c_kl_lambda,
            optimizer=vae_optimizer,
            connected_weights=connected_weights,
            categorical_latent_embedding_dropout=categorical_latent_embedding_dropout,
            mixture_latent_mu_embedding_dropout=mixture_latent_mu_embedding_dropout,
            mixture_latent_var_embedding_dropout=mixture_latent_var_embedding_dropout,
            mixture_posterior_mu_dropout=mixture_posterior_mu_dropout,
            mixture_posterior_var_dropout=mixture_posterior_var_dropout,
            recon_dropouut=recon_dropouut,
            latent_fixed_var=latent_fixed_var,
        )
        self.descriminator = SigmoidEncoder(
            latent_dimension=1,
            embedding_dimensions=descriminator_dimensions,
            var_scope=self.v_name("graph_descriminator"),
            bn_before=bn_before,
            bn_after=bn_after,
            epsilon=0.0,
            embedding_kernel_initializer=descr_embedding_kernel_initializer,
            embedding_bias_initializer=descr_embedding_bias_initializer,
            latent_kernel_initialiazer=descr_latent_kernel_initialiazer,
            latent_bias_initializer=descr_latent_bias_initializer,
        )

        self.encoder_vars = (
            self.gmvae.marginal_autoencoder.graphs_qz_g_xy.trainable_variables
            + list(self.gmvae.graph_qy_g_x.trainable_variables)
        )
        self.decoder_vars = (
            self.gmvae.marginal_autoencoder.graphs_px_g_zy.trainable_variables
        )
        self.gan_vars = self.descriminator.trainable_variables

    def increment_cooling(self):
        self.cooling_distance += 1

    @tf.function
    def sample_one(
        self, x, training=False, temperature=1.0, beta_z=1.0, beta_y=1.0
    ):

        # Sample from the generator
        gmvaeres = self.gmvae.sample_one(x, training, temperature)

        # get the prob from the descriminator for the true distribution
        (
            descr_true__sample,
            descr_true__logprob,
            descr_true__prob,
        ) = self.descriminator.call(x, training)

        # get the prob from the descriminator for the true distribution
        (
            descr_gen__sample,
            descr_gen__logprob,
            descr_gen__prob,
        ) = self.descriminator.call(
            gmvaeres["autoencoder"]["px_g_zy__sample"], training
        )

        # desciminator loss
        descriminator_entropy = descr_true__logprob + tf.math.log(
            1 - descr_gen__prob
        )

        output = {
            "gmvae": gmvaeres,
            "desc_true__sample": descr_true__sample,
            "descr_true__logprob": descr_true__logprob,
            "descr_true__prob": descr_true__prob,
            "descr_gen__sample": descr_gen__sample,
            "descr_gen__logprob": descr_gen__logprob,
            "descr_gen__prob": descr_gen__prob,
            "descr_ent": descriminator_entropy,
        }

        return output

    @tf.function(experimental_relax_shapes=True)
    def sample(self, samples, x, training=False, temperature=1.0):
        with tf.device("/gpu:0"):
            result = [
                self.sample_one(x, training, temperature)
                for j in range(samples)
            ]
        return result

    @tf.function(experimental_relax_shapes=True)
    def monte_carlo_estimate(
        self, samples, x, training=False, temperature=1.0
    ):
        return mc_stack_mean_dict(
            self.sample(samples, x, training, temperature)
        )

    @tf.function
    def call(self, x, training=False, samples=1, temperature=1.0):
        output = self.monte_carlo_estimate(samples, x, training, temperature)
        return output

    @tf.function
    def latent_sample(self, inputs, training=False, samples=1):
        return self.gmvae.latent_sample(inputs, training, samples)

    @tf.function  # (autograph=False)
    def entropy_fn(self, inputs, training=False, samples=1, temperature=1.0):
        # unclear why tf.function  doesnt work to decorate this
        output = self.call(
            inputs, training=training, samples=samples, temperature=temperature
        )
        # return output
        return (
            output["gmvae"]["recon"],
            output["gmvae"]["z_entropy"],
            output["gmvae"]["y_entropy"],
            output["descr_ent"],
        )

    @tf.function
    def loss_fn(
        self,
        inputs,
        training=False,
        samples=1,
        temperature=1.0,
        beta_z=1.0,
        beta_y=1.0,
        beta_d=1.0,
    ):

        recon, z_ent, y_ent, d_ent = self.entropy_fn(
            inputs, training, samples, temperature
        )

        elbo = recon + beta_z * z_ent + beta_y * y_ent

        enc_loss = -elbo
        recgan_loss = -(recon + beta_z * z_ent - beta_d * d_ent)
        gan_loss = -d_ent * beta_d

        output = [enc_loss, recgan_loss, gan_loss]
        return output

    @tf.function  # (autograph=False)
    def predict(self, x, training=False):

        x = tf.cast(x, dtype=self.dtype)
        x = tf.cast(
            tf.where(tf.math.is_nan(x), tf.ones_like(x) * 0.0, x),
            dtype=self.dtype,
        )

        qy_g_x__logit, qy_g_x__prob = self.gmvae.graph_qy_g_x(
            x, training=training
        )
        return qy_g_x__prob

    @tf.function  # (autograph=False)
    def train_step(
        self,
        inputs,
        temperature=1.0,
        beta_z: float = 1.0,
        beta_y: float = 1.0,
        beta_d: float = 1.0,
        training: bool = True,
        samples: int = 1,
        gradient_clip: Optional[float] = None,
        weights: Optional[Tensor] = None,
    ):

        with tf.device("/gpu:0"):

            # Persistent gradient tape is breaking the results
            # with tf.GradientTape(persistent=False) as tape:
            #    #weights = 1.0 if weights is None else weights
            #
            #    enc_loss, recgan_loss, gan_loss = self.loss_fn(
            #        inputs, training, samples, temperature,
            #        beta_z, beta_y,beta_d
            #    )

            # Replace with multiple forward passes for the moment
            with tf.GradientTape(persistent=False) as tape:
                enc_loss, recgan_loss, gan_loss = self.loss_fn(
                    inputs,
                    training,
                    samples,
                    temperature,
                    beta_z,
                    beta_y,
                    beta_d,
                )
                enc_grad = tape.gradient(enc_loss, self.encoder_vars)

            with tf.GradientTape(persistent=False) as tape:
                enc_loss, recgan_loss, gan_loss = self.loss_fn(
                    inputs,
                    training,
                    samples,
                    temperature,
                    beta_z,
                    beta_y,
                    beta_d,
                )
                dec_grad = tape.gradient(recgan_loss, self.decoder_vars)

            with tf.GradientTape(persistent=False) as tape:
                enc_loss, recgan_loss, gan_loss = self.loss_fn(
                    inputs,
                    training,
                    samples,
                    temperature,
                    beta_z,
                    beta_y,
                    beta_d,
                )
                gan_grad = tape.gradient(gan_loss, self.gan_vars)

        # Clipping
        grad_func = lambda gradients, gradient_clip: [
            None
            if gradient is None
            else tf.clip_by_value(gradient, -gradient_clip, gradient_clip)
            if gradient_clip is not None
            else gradient
            for gradient in gradients
        ]

        enc_grad = grad_func(enc_grad, gradient_clip)
        dec_grad = grad_func(dec_grad, gradient_clip)
        gan_grad = grad_func(gan_grad, gradient_clip)

        # update gradients as per: https://arxiv.org/pdf/1512.09300.pdf
        # NOTE: We do not perform the invariant transormfation to the
        # reconstruction likelihood for the endoder
        self.vae_optimizer.apply_gradients(zip(enc_grad, self.encoder_vars))
        self.dec_optimizer.apply_gradients(zip(dec_grad, self.decoder_vars))
        self.gan_optimizer.apply_gradients(zip(gan_grad, self.gan_vars))

        return gan_loss, enc_loss, recgan_loss, gan_loss

import tensorflow as tf
from tensorflow.python.eager import context
import numpy as np
import datetime

from deeper.probability_layers.gumble_softmax import GumbleSoftmaxLayer
from deeper.probability_layers.normal import RandomNormalEncoder, lognormal_kl
from deeper.layers.binary import SigmoidEncoder
from deeper.layers.categorical import CategoricalEncoder
from deeper.utils.scope import Scope
from deeper.models.gmvae.marginalautoencoder import MarginalAutoEncoder
from deeper.utils.function_helpers.decorators import inits_args
from deeper.utils.sampling import mc_stack_mean_dict

tfk = tf.keras

Model = tfk.Model


class Gmvae(Model, Scope):
    @inits_args
    def __init__(
        self,
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
        var_scope="gmvae",
        cat_embedding_kernel_initializer="glorot_uniform",
        cat_embedding_bias_initializer="zeros",
        cat_latent_kernel_initialiazer="glorot_uniform",
        cat_latent_bias_initializer="zeros",
        latent_mu_embedding_kernel_initializer="glorot_uniform",
        latent_mu_embedding_bias_initializer="zeros",
        latent_mu_latent_kernel_initialiazer="glorot_uniform",
        latent_mu_latent_bias_initializer="zeros",
        latent_var_embedding_kernel_initializer="glorot_uniform",
        latent_var_embedding_bias_initializer="zeros",
        latent_var_latent_kernel_initialiazer="glorot_uniform",
        latent_var_latent_bias_initializer="zeros",
        posterior_mu_embedding_kernel_initializer="glorot_uniform",
        posterior_mu_embedding_bias_initializer="zeros",
        posterior_mu_latent_kernel_initialiazer="glorot_uniform",
        posterior_mu_latent_bias_initializer="zeros",
        posterior_var_embedding_kernel_initializer="glorot_uniform",
        posterior_var_embedding_bias_initializer="zeros",
        posterior_var_latent_kernel_initialiazer="glorot_uniform",
        posterior_var_latent_bias_initializer="zeros",
        recon_embedding_kernel_initializer="glorot_uniform",
        recon_embedding_bias_initializer="zeros",
        recon_latent_kernel_initialiazer="glorot_uniform",
        recon_latent_bias_initializer="zeros",
        z_kl_lambda=1.0,
        c_kl_lambda=1.0,
        optimizer=tf.keras.optimizers.SGD(0.001),
        connected_weights=True,
        categorical_latent_embedding_dropout=0.0,
        mixture_latent_mu_embedding_dropout=0.0,
        mixture_latent_var_embedding_dropout=0.0,
        mixture_posterior_mu_dropout=0.0,
        mixture_posterior_var_dropout=0.0,
        recon_dropouut=0.0,
        latent_fixed_var=None,
    ):

        # instatiate
        Model.__init__(self)
        Scope.__init__(self, var_scope)

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

        self.cooling_distance = tf.constant(0.0, dtype=self.dtype)

        if cat_latent_bias_initializer is None:
            cat_latent_bias_initializer = tf.initializers.constant(
                np.log((1 / self.components) / (1 - 1 / self.components))
            )

        with tf.name_scope("categorical"):
            self.graph_qy_g_x = CategoricalEncoder(
                latent_dimension=components,
                embedding_dimensions=embedding_dimensions,
                embedding_activation=embedding_activations,
                var_scope=self.v_name("categorical_encoder"),
                bn_before=bn_before,
                bn_after=bn_after,
                epsilon=categorical_epsilon,
                embedding_kernel_initializer=cat_embedding_kernel_initializer,
                embedding_bias_initializer=cat_embedding_bias_initializer,
                latent_kernel_initialiazer=cat_latent_kernel_initialiazer,
                latent_bias_initializer=cat_latent_bias_initializer,
                embedding_dropout=categorical_latent_embedding_dropout,
            )
            self.graph_qy_g_x_ohe = GumbleSoftmaxLayer()

        self.marginal_autoencoder = MarginalAutoEncoder(
            self.input_dimension,
            self.mem_dim,
            self.mem_lat,
            kind=self.kind,
            var_scope=self.v_name("marginal_autoencoder"),
            latent_epsilon=self.latent_epsilon,
            reconstruction_epsilon=self.reconstruction_epsilon,
            embedding_activations=self.mem_act,
            latent_prior_epsilon=latent_prior_epsilon,
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
            connected_weights=connected_weights,
            latent_mu_embedding_dropout=mixture_latent_mu_embedding_dropout,
            latent_var_embedding_dropout=mixture_latent_var_embedding_dropout,
            posterior_mu_dropout=mixture_posterior_mu_dropout,
            posterior_var_dropout=mixture_posterior_var_dropout,
            recon_dropouut=recon_dropouut,
            latent_fixed_var=latent_fixed_var,
        )

        self.optimizer = optimizer

    def increment_cooling(self):
        self.cooling_distance += 1.0

    @tf.function
    def sample_one(self, inputs, training=False, temperature=1.0):

        x = inputs
        # x = tf.cast(inputs, dtype=self.dtype)
        # x = tf.cast(
        #    tf.where(tf.math.is_nan(x), tf.ones_like(x) * 0.0, x),
        #    dtype=self.dtype,
        # )

        py = tf.cast(
            tf.fill(
                (tf.shape(x)[0], self.components),
                1 / self.components,
                name="prob",
            ),
            x.dtype,
        )

        qy_g_x__logit, qy_g_x__prob = self.graph_qy_g_x(x, training=training)
        qy_g_x_ohe = self.graph_qy_g_x_ohe(qy_g_x__logit, temperature)

        mres = self.marginal_autoencoder(x, qy_g_x_ohe, training)

        # Losses
        # reconstruction
        recon = mres["px_g_zy__logprob"]

        # z_entropy
        z_entropy = mres["pz_g_y__logprob"] - mres["qz_g_xy__logprob"]

        # y_entropy
        # E_q [log(p/q)] = sum q (log_p - log_q)
        y_entropy = tf.reduce_sum(
            qy_g_x__prob * tf.math.log(py), -1
        ) + tf.nn.softmax_cross_entropy_with_logits(
            logits=qy_g_x__logit, labels=qy_g_x__prob
        )

        # elbo
        elbo = (
            recon + self.z_kl_lambda * z_entropy + self.c_kl_lambda * y_entropy
        )

        output = {
            "py": py,
            "qy_g_x__logit": qy_g_x__logit,
            "qy_g_x__prob": qy_g_x__prob,
            "qy_g_x_ohe": qy_g_x_ohe,
            "recon": recon,
            "z_entropy": z_entropy,
            "y_entropy": y_entropy,
            "autoencoder": mres,
        }

        return output

    @tf.function
    def sample(self, samples, x, training=False, temperature=1.0):
        result = [
            self.sample_one(x, training, temperature) for j in range(samples)
        ]
        return result

    @tf.function
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
        outputs = self.call(inputs, training=training, samples=samples)
        latent = outputs["autoencoder"]["px_g_zy__sample"]
        return latent

    def call_even(self, x, training=False, samples=1):
        pass

    @tf.function
    def entropy_fn(self, inputs, training=False, samples=1, temperature=1.0):
        # unclear why tf.function  doesnt work to decorate this
        output = self.call(
            inputs, training=training, samples=samples, temperature=temperature
        )
        # return output
        return output["recon"], output["z_entropy"], output["y_entropy"]

    @tf.function
    def elbo(
        self,
        inputs,
        training=False,
        samples=1,
        temperature=1.0,
        beta_z=1.0,
        beta_y=1.0,
    ):
        recon, z_entropy, y_entropy = self.entropy_fn(
            inputs, training, samples, temperature
        )
        return recon + beta_z * z_entropy + beta_y * y_entropy

    @tf.function
    def loss_fn(
        self,
        inputs,
        training=False,
        samples=1,
        temperature=1.0,
        beta_z=1.0,
        beta_y=1.0,
    ):
        loss = -tf.reduce_mean(
            self.elbo(inputs, training, samples, temperature, beta_z, beta_y)
        )
        return loss

    def even_mixture_loss(self, inputs, training=False, samples=1, beta_z=1.0):
        pass

    @tf.function
    def train_step(
        self,
        x,
        samples=1,
        tenorboard=False,
        batch=False,
        temperature=1.0,
        beta_z=1.0,
        beta_y=1.0,
    ):

        if tenorboard:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = "logs/gradient_tape/train"

            writer = tf.summary.create_file_writer(train_log_dir)

        # for x in dataset:
        # Tensorflow dataset is iterable in eager mode
        # loss = self.loss_fn(x, True, samples, temperature, beta_z, beta_y)
        with tf.GradientTape() as tape:
            loss = self.loss_fn(x, True, samples, temperature, beta_z, beta_y)

        # Update ops for batch normalization
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):

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

    # @tf.function
    def pretrain_step(self, x, samples=1, batch=False, beta_z=1.0):
        # for x in dataset:
        # Tensorflow dataset is iterable in eager mode
        target_vars = [
            v
            for v in self.trainable_variables
            if "gmvae/marginal_autoencoder" in v.name
        ]
        with tf.device("/gpu:0"):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                # tape.watch(target_vars)

                if batch:
                    loss = tf.reduce_mean(
                        self.even_mixture_loss(x, True, samples, beta_z)
                    )
                else:
                    loss = self.even_mixture_loss(x, True, samples)
                # Update ops for batch normalization
                # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # with tf.control_dependencies(update_ops):

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

                self.optimizer.apply_gradients(
                    zip(gradients, self.trainable_variables)
                )

    @tf.function
    def predict(self, x, training=False):

        # x = tf.cast(x, dtype=self.dtype)
        # x = tf.cast(
        #    tf.where(tf.math.is_nan(x), tf.ones_like(x) * 0.0, x),
        #    dtype=self.dtype,
        # )

        qy_g_x__logit, qy_g_x__prob = self.graph_qy_g_x(x, training=training)
        return qy_g_x__prob

    def cluster_loss(self, x, y, training=False):
        qy_g_x__logit, qy_g_x__prob = self.graph_qy_g_x(x, training=training)
        # nent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #    logits=qy_g_x__logit,
        #    labels=y
        # ), axis=-1)

        nent = -tf.add_n(
            [
                y[:, i]
                * (tf.math.log(qy_g_x__prob[:, i]) - tf.math.log(y[:, i]))
                for i in range(self.k)
            ]
        )

        return nent

    def pretrain_categories_step(self, x, y, samples=1):
        y = tf.clip_by_value(y, 0.05, 0.95)
        with tf.device("/gpu:0"):
            with tf.GradientTape() as tape:
                loss = tf.add(
                    tf.reduce_mean(self.cluster_loss(x, y, True)),
                    tf.reduce_mean(
                        self.loss_fn(x, True, samples, temperature=0.6)
                    ),
                )

        with tf.device("/gpu:0"):
            gradients = tape.gradient(loss, self.trainable_variables)
            # Clipping
            gradients = [
                None
                if gradient is None
                else tf.clip_by_value(
                    gradient, -self.gradient_clip, self.gradient_clip
                )
                for gradient in gradients
            ]
        with tf.device("/gpu:0"):
            self.optimizer.apply_gradients(
                zip(gradients, self.trainable_variables)
            )

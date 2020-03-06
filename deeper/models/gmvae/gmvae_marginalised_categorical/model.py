import tensorflow as tf
from tensorflow.python.eager import context
import numpy as np
import datetime

from deeper.ops.distance import kl_divergence
from deeper.layers.binary import SigmoidEncoder
from deeper.layers.categorical import CategoricalEncoder
from deeper.probability_layers.gumble_softmax import GumbleSoftmaxLayer
from deeper.probability_layers.normal import RandomNormalEncoder, lognormal_kl
from deeper.utils.scope import Scope
from deeper.utils.function_helpers.decorators import inits_args
from deeper.utils.function_helpers.collectors import get_local_tensors
from deeper.utils.sampling import mc_stack_mean_dict
from deeper.models.gmvae.marginalautoencoder import MarginalAutoEncoder

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

        self.cooling_distance = 0

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

        # self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.optimizer = optimizer

    def increment_cooling(self):
        self.cooling_distance += 1

    # @tf.function
    def sample_one(self, inputs, training=False):
        x = tf.cast(inputs, dtype=self.dtype)

        x = tf.cast(
            tf.where(tf.math.is_nan(x), tf.ones_like(x) * 0.0, x),
            dtype=self.dtype,
        )
        # x = tf.cast(tf.greater(tf.cast(x,tf.float32), tf.random_uniform(tf.shape(x), 0, 1)), tf.float32)
        # Add random binarizer

        y_ = tf.cast(
            tf.fill(tf.stack([tf.shape(x)[0], self.components]), 0.0), x.dtype
        )
        py = tf.cast(
            tf.fill(
                (tf.shape(x)[0], self.components),
                1 / self.components,
                name="prob",
            ),
            x.dtype,
        )
        qy_g_x__logit, qy_g_x__prob = self.graph_qy_g_x(x, training=training)

        mres = {}  # marginal encoder results
        for i in range(self.components):
            with tf.name_scope("mixture_{}".format(i)):
                y = tf.add(
                    y_,
                    tf.constant(
                        np.eye(self.components)[i],
                        dtype=x.dtype,
                        name="y_one_hot_{}".format(i),
                    ),
                    name="hot_at_{}".format(i),
                )
                mres[i] = self.marginal_autoencoder(x, y, training)

        # Losses
        # reconstruction
        recon = tf.add_n(
            [
                #tf.math.exp(tf.nn.log_softmax(qy_g_x__logit[:, i]))
                qy_g_x__prob[:, i] 
                * (mres[i]["px_g_zy__logprob"])
                for i in range(self.components)
            ]
        )

        # z_entropy
        z_entropy = tf.add_n(
            [
                tf.math.exp(tf.nn.log_softmax(qy_g_x__logit[:, i]))
                #qy_g_x__prob[:, i]
                * (mres[i]["pz_g_y__logprob"] - mres[i]["qz_g_xy__logprob"])
                for i in range(self.components)
            ]
        )

        # y_entropy
        # E_q [log(p/q)] = sum q (log_p - log_q)
        #y_entropy = tf.reduce_sum(
        #    tf.math.exp(tf.nn.log_softmax(qy_g_x__logit)) * tf.math.log(py), -1
        #) + tf.nn.softmax_cross_entropy_with_logits(
        #    logits=qy_g_x__logit, labels=qy_g_x__prob
        #)
        y_entropy = tf.reduce_sum(
            tf.math.exp(tf.nn.log_softmax(qy_g_x__logit))
             * (tf.math.log(py) - tf.nn.log_softmax(qy_g_x__logit)), -1)

        # elbo
        elbo = (
            recon + self.z_kl_lambda * z_entropy + self.c_kl_lambda * y_entropy
        )

        output = {
            "py": py,
            "qy_g_x__logit": qy_g_x__logit,
            "qy_g_x__prob": qy_g_x__prob,
            "recon": recon,
            "z_entropy": z_entropy,
            "y_entropy": y_entropy,
            "autoencoder": mres,
        }
        # return {**get_local_tensors(locals()), **{'autoencoder': mres}}
        return output

    @tf.function(experimental_relax_shapes=True)
    def sample(self, samples, x, training=False):
        with tf.device("/gpu:0"):
            result = [self.sample_one(x, training) for j in range(samples)]
        return result

    @tf.function(experimental_relax_shapes=True)
    def monte_carlo_estimate(self, samples, x, training=False):
        return mc_stack_mean_dict(self.sample(samples, x, training))

    @tf.function
    def call(self, x, training=False, samples=1):
        output = self.monte_carlo_estimate(samples, x, training)
        return output

    @tf.function
    def latent_sample(self, inputs, training=False, samples=1):
        vals = self.call(inputs, training=training, samples=samples)

        latent = tf.add_n(
            [
                vals["qy_g_x__prob"][:, i, None]
                * (vals["autoencoder"][i]["qz_g_xy__sample"])
                for i in range(self.components)
            ]
        )

        return latent

    @tf.function
    def entropy_fn(self, inputs, training=False, samples=1):
        output = self.call(inputs, training=training, samples=samples)
        return output["recon"], output["z_entropy"], output["y_entropy"]

    @tf.function
    def elbo(self, inputs, training=False, samples=1, beta_z=1.0, beta_y=1.0):
        recon, z_entropy, y_entropy = self.entropy_fn(
            inputs, training, samples
        )
        return recon + beta_z * z_entropy + beta_y * y_entropy

    @tf.function
    def loss_fn(
        self, inputs, training=False, samples=1, beta_z=1.0, beta_y=1.0
    ):
        return -self.elbo(inputs, training, samples, beta_z, beta_y)


    @tf.function
    def loss_fn_even(
        self, inputs, training=False, samples=1, beta_z=1.0, beta_y=1.0
    ):
        vals = self.call(inputs, training=training, samples=samples)
        recon = tf.add_n([
            vals["autoencoder"][i]["px_g_zy__logprob"] 
            for i in range(self.components)
        ]) / self.components

        return recon


    @tf.function
    def loss_fn_with_known_clusters(
        self, x, y, training=False, samples=1, beta_z=1.0, beta_y=1.0
    ):

        even_elbo = self.elbo(x, training, samples, beta_z, 0.0)
        cluster_loss = self.cluster_loss(x, y, True)

        return -even_elbo + beta_y * cluster_loss


    @tf.function
    def train_base(
        self, 
        loss_fn, 
        x,
        samples=1,
        tenorboard=False,
        batch=False,
        beta_z=1.0,
        beta_y=1.0,
    ):

        if tenorboard:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = "logs/gradient_tape/train"

            writer = tf.summary.create_file_writer(train_log_dir)

        # for x in dataset:
        # Tensorflow dataset is iterable in eager mode
        #with tf.device("/gpu:0"):
        with tf.GradientTape() as tape:
            if batch:
                loss = tf.reduce_mean(
                    loss_fn(x, True, samples, beta_z, beta_y)
                )
            else:
                loss = loss_fn(x, True, samples, beta_z, beta_y)
            # Update ops for batch normalization
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):

        #with tf.device("/gpu:0"):
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

        #with tf.device("/gpu:0"):
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )
    


    @tf.function
    def train_step(
        self,
        x,
        samples=1,
        tenorboard=False,
        batch=False,
        beta_z=1.0,
        beta_y=1.0,
    ):
        self.train_base(self.loss_fn, x, samples, tenorboard, batch, beta_z, beta_y)

    @tf.function
    def pretrain_step(self, x, samples=1, batch=False, beta_z=1.0, beta_y=1.0,):
        self.train_base(self.loss_fn_even, x, samples, False, batch, beta_z, beta_y)

    @tf.function
    def predict(self, x, training=False):
        qy_g_x__logit, qy_g_x__prob = self.graph_qy_g_x(x, training=training)
        return qy_g_x__prob

    @tf.function
    def cluster_loss(self, x, y, training=False):
        qy_g_x__logit, qy_g_x__prob = self.graph_qy_g_x(x, training)
        nent = -tf.add_n(
            [
                y[:, i]
                * (tf.math.log(qy_g_x__prob[:, i]) - tf.math.log(y[:, i]))
                for i in range(self.components)
            ]
        )
        return nent

    def pretrain_categories_step(
        self, x, y, samples=1, beta_z=1.0, beta_y=1.0
    ):
        # y = tf.clip_by_value(y, 0.05, 0.95)

        # Compute gradients
        with tf.device("/gpu:0"):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(
                    self.loss_fn_with_known_clusters(
                        x, y, True, samples, beta_z, beta_y
                    )
                )

        # backprop and clippint
        with tf.device("/gpu:0"):
            gradients = tape.gradient(loss, self.trainable_variables)
            gradients = [
                None
                if gradient is None
                else tf.clip_by_value(
                    gradient, -self.gradient_clip, self.gradient_clip
                )
                if self.gradient_clip is not None
                else gradient
                for gradient in gradients
            ]

        # update
        with tf.device("/gpu:0"):
            self.optimizer.apply_gradients(
                zip(gradients, self.trainable_variables)
            )

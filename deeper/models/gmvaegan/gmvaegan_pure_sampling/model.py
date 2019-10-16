import tensorflow as tf
from tensorflow.python.eager import context
import numpy as np
import datetime

from deeper.models.gmvae.gmvae_pure_sampling.model import Gmvae
from deeper.probability_layers.gumble_softmax import GumbleSoftmaxLayer
from deeper.probability_layers.normal import RandomNormalEncoder
from deeper.layers.binary import SigmoidEncoder
from deeper.layers.categorical import CategoricalEncoder
from deeper.utils.scope import Scope

tfk = tf.keras

Model = tfk.Model

class GmvaeGan(Model, Scope):
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
        var_scope='gmvaegan',

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

        optimizer=tf.keras.optimizers.SGD(0.001),

        connected_weights=True,
    ):
        self.var_scope = 'gmvaegan'
        self.kind = kind
        self.k = components
        self.in_dim = input_dimension
        self.em_dim = embedding_dimensions
        self.la_dim = latent_dimensions
        self.em_act = embedding_activations

        self.mem_dim = (
            mixture_embedding_dimensions 
            if mixture_embedding_dimensions is not None
            else self.em_dim
        )
        self.mem_act = (
            mixture_embedding_activations
            if mixture_embedding_activations is not None
            else self.em_act
        )
        self.mem_lat = (
            mixture_latent_dimensions
            if mixture_latent_dimensions is not None 
            else self.la_dim
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
        self.optimizer = optimizer

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
            var_scope=self.v_name('gmvae'),
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
            optimizer=optimizer,
            connected_weights=connected_weights
        )
        self.descriminator = SigmoidEncoder(
            latent_dimension=1, 
            embedding_dimensions=descriminator_dimensions, 
            var_scope=self.v_name('graph_descriminator'),
            bn_before=bn_before,
            bn_after=bn_after,
            epsilon=0.0,
            embedding_kernel_initializer=descr_embedding_kernel_initializer,
            embedding_bias_initializer=descr_embedding_bias_initializer,
            latent_kernel_initialiazer=descr_latent_kernel_initialiazer,
            latent_bias_initializer=descr_latent_bias_initializer,
        )


    def sample_one(self, x, training=False, temperature=1.0):

        y_ = tf.cast(
            tf.fill(tf.stack([tf.shape(x)[0], self.k]), 0.0),
            dtype=x.dtype
        )
        py = tf.cast(tf.fill(tf.shape(y_), 1 / self.k, name="prob"), x.dtype)

        # Sample from the generator
        (
            qy_g_x__prob,
            qz_g_xy__sample,
            qz_g_xy__logprob,
            qz_g_xy__prob,
            pz_g_y__sample,
            pz_g_y__logprob,
            pz_g_y__prob,
            dkl_z_g_xy,
            px_g_zy__sample,
            px_g_zy__logprob,
            px_g_zy__prob,
        ) = self.gmvae.sample_one(x, training, temperature)


        # gmvae Loss parameters
        # y_entropy
        dkl_y = y_entropy = tf.reduce_sum(
            qy_g_x__prob * (tf.math.log(py) - tf.math.log(qy_g_x__prob)),
            axis=-1,
        )

        elbo = px_g_zy__logprob + dkl_z_g_xy + dkl_y
        gmvae_loss = - elbo


        # get the prob from the descriminator for the true distribution
        (
            descr_true__sample,
            descr_true__logprob,
            descr_true__prob
        ) = self.descriminator.call(x, training)

        # get the prob from the descriminator for the true distribution
        (
            descr__sample,
            descr__logprob,
            descr__prob
        ) = self.descriminator.call(px_g_zy__sample, training)

        # desciminator loss
        descriminator_entropy = (
            descr_true__logprob + tf.math.log(1-descr__prob)
        )

        loss = tf.reduce_mean(gmvae_loss - descriminator_entropy, axis=-1)

        return (
            py,
            qy_g_x__prob,
            qz_g_xy__sample,
            qz_g_xy__logprob,
            qz_g_xy__prob,
            pz_g_y__sample,
            pz_g_y__logprob,
            pz_g_y__prob,
            dkl_z_g_xy,
            px_g_zy__sample,
            px_g_zy__logprob,
            px_g_zy__prob,
            dkl_y,
            elbo,
            gmvae_loss,
            descr_true__sample,
            descr_true__logprob,
            descr_true__prob,
            descr__sample,
            descr__logprob,
            descr__prob,
            descriminator_entropy,
            loss
        )


    @tf.function(experimental_relax_shapes=True)
    def sample(self, samples, x, training=False, temperature=1.0):
        with tf.device("/gpu:0"):
            result = [self.sample_one(x, training, temperature) for j in range(samples)]
            result_pivot = list(zip(*result))
        return result_pivot


    @staticmethod
    @tf.function
    def mc_stack_mean(x):
        return tf.reduce_sum(tf.stack(x, 0), 0) / len(x)


    @tf.function(experimental_relax_shapes=True)
    def monte_carlo_estimate(self, samples, x, training=False, temperature=1.0):
        return [
            self.mc_stack_mean(z)
            for z in self.sample(samples, x, training=False, temperature=temperature)
        ]


    @tf.function 
    def call(self, x, training=False, samples=1, temperature=1.0):
         return self.monte_carlo_estimate( samples, x, training, temperature)


    @tf.function
    def loss_fn(self, inputs, training=False, samples=1, temperature=1.0):
        (
            py,
            qy_g_x__prob,
            qz_g_xy__sample,
            qz_g_xy__logprob,
            qz_g_xy__prob,
            pz_g_y__sample,
            pz_g_y__logprob,
            pz_g_y__prob,
            dkl_z_g_xy,
            px_g_zy__sample,
            px_g_zy__logprob,
            px_g_zy__prob,
            dkl_y,
            elbo,
            gmvae_loss,
            descr_true__sample,
            descr_true__logprob,
            descr_true__prob,
            descr__sample,
            descr__logprob,
            descr__prob,
            descriminator_entropy,
            loss
        ) = self.call(inputs, training, samples, temperature)

        return loss
    
    def entropy_fn(self, inputs, training=False, samples=1, temperature=1.0):
        (
            py,
            qy_g_x__prob,
            qz_g_xy__sample,
            qz_g_xy__logprob,
            qz_g_xy__prob,
            pz_g_y__sample,
            pz_g_y__logprob,
            pz_g_y__prob,
            dkl_z_g_xy,
            px_g_zy__sample,
            px_g_zy__logprob,
            px_g_zy__prob,
            dkl_y,
            elbo,
            gmvae_loss,
            descr_true__sample,
            descr_true__logprob,
            descr_true__prob,
            descr__sample,
            descr__logprob,
            descr__prob,
            descriminator_entropy,
            loss
        ) = self.call(inputs, training, samples, temperature)

        return px_g_zy__logprob, dkl_z_g_xy, dkl_y, descriminator_entropy

    @tf.function
    def train_step(
        self, 
        x, 
        samples=1, 
        tenorboard=False, 
        batch=False, 
        temperature=1.0
    ):

        if tenorboard:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/gradient_tape/train'

            writer = tf.summary.create_file_writer(train_log_dir)


        # for x in dataset:
        # Tensorflow dataset is iterable in eager mode
        with tf.device("/gpu:0"):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(self.loss_fn(x, True, samples, temperature))
            # Update ops for batch normalization
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):

        with tf.device("/gpu:0"):
            gradients = tape.gradient(loss, self.trainable_variables)

            # Clipping
            gradients = [
                None
                if gradient is None
                else tf.clip_by_value(
                    gradient, -self.gradient_clip, self.gradient_clip
                )
                #else tf.clip_by_norm(
                #    gradient, self.gradient_clip
                #)
                for gradient in gradients
            ]

            if tenorboard:
                with writer.as_default():
                    for gradient, variable in zip(gradients, self.trainable_variables):
                        steps = steps + 1
                        tf.summary.experimental.set_step(steps)
                        stp = tf.summary.experimental.get_step()
                        tf.summary.histogram("gradients/" + variable.name, tf.nn.l2_normalize(gradient), step=stp)
                        tf.summary.histogram("variables/" + variable.name, tf.nn.l2_normalize(variable), step=stp)
                    writer.flush()

        with tf.device("/gpu:0"):
            self.optimizer.apply_gradients(
                zip(gradients, self.trainable_variables)
            )

    @tf.function
    def predict(self, x, training=False):
        qy_g_x__logit, qy_g_x__prob = self.gmvae.graph_qy_g_x(x, training)
        return qy_g_x__prob

    def increment_cooling(self):
        self.cooling_distance += 1


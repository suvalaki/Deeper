import tensorflow as tf
from tensorflow.python.eager import context
import numpy as np
import datetime

from deeper.probability_layers.gumble_softmax import GumbleSoftmaxLayer
from deeper.probability_layers.normal import (
    RandomNormalEncoder, RandomStandardNormalEncoder
)
from deeper.layers.binary import SigmoidEncoder
from deeper.layers.categorical import CategoricalEncoder
from deeper.utils.scope import Scope

tfk = tf.keras

Model = tfk.Model

class MarginalAutoEncoder(Model, Scope):
    def __init__(
        self, 
        input_dimension, 
        embedding_dimensions, 
        latent_dim, 
        embedding_activations=tf.nn.tanh,
        kind="binary",
        var_scope='marginal_autoencoder',
        bn_before=False,
        bn_after=False,
        latent_epsilon=0.0,
        latent_prior_epsilon=0.0,
        reconstruction_epsilon=0.0,

        latent_mu_embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        latent_mu_embedding_bias_initializer=tf.initializers.zeros(),
        latent_mu_latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        latent_mu_latent_bias_initializer=tf.initializers.zeros(),

        latent_var_embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        latent_var_embedding_bias_initializer=tf.initializers.zeros(),
        latent_var_latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        latent_var_latent_bias_initializer=tf.initializers.ones(),

        posterior_mu_embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        posterior_mu_embedding_bias_initializer=tf.initializers.zeros(),
        posterior_mu_latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        posterior_mu_latent_bias_initializer=tf.initializers.zeros(),

        posterior_var_embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        posterior_var_embedding_bias_initializer=tf.initializers.zeros(),
        posterior_var_latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        posterior_var_latent_bias_initializer=tf.initializers.ones(),

        recon_embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        recon_embedding_bias_initializer=tf.initializers.zeros(),
        recon_latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        recon_latent_bias_initializer=tf.initializers.zeros(),

    ):
        Model.__init__(self)
        Scope.__init__(self, var_scope)
        self.in_dim = input_dimension
        self.la_dim = latent_dim
        self.em_dim = embedding_dimensions
        self.kind = kind
        self.bn_before = bn_before
        self.bn_after = bn_after
        self.lat_eps = latent_epsilon
        self.lat_p_eps = latent_prior_epsilon
        self.rec_eps = reconstruction_epsilon


        with tf.name_scope('graph_qz_g_xy'):
            self.graphs_qz_g_xy = RandomNormalEncoder(
                latent_dimension=self.la_dim, 
                embedding_dimensions=self.em_dim, 
                var_scope=self.v_name('graph_qz_g_xy'),
                bn_before=self.bn_before,
                bn_after=self.bn_after,
                epsilon=self.lat_eps,

                embedding_mu_kernel_initializer=latent_mu_embedding_kernel_initializer,
                embedding_mu_bias_initializer=latent_mu_embedding_bias_initializer,
                latent_mu_kernel_initialiazer=latent_mu_latent_kernel_initialiazer,
                latent_mu_bias_initializer=latent_mu_latent_bias_initializer,

                embedding_var_kernel_initializer=latent_var_embedding_kernel_initializer,
                embedding_var_bias_initializer=latent_var_embedding_bias_initializer,
                latent_var_kernel_initialiazer=latent_var_latent_kernel_initialiazer,
                latent_var_bias_initializer=latent_var_latent_bias_initializer,
            )
        with tf.name_scope('graph_pz_g_y'):
            self.graphs_pz_g_y = RandomStandardNormalEncoder(
                latent_dimension=self.la_dim, 
                embedding_dimensions=[], 
                var_scope=self.v_name('graph_pz_g_y'),
                bn_before=self.bn_before,
                bn_after=self.bn_after,
                epsilon=self.lat_p_eps,

                embedding_mu_kernel_initializer=posterior_mu_embedding_kernel_initializer,
                embedding_mu_bias_initializer=posterior_mu_embedding_bias_initializer,
                latent_mu_kernel_initialiazer=posterior_mu_latent_kernel_initialiazer,
                latent_mu_bias_initializer=posterior_mu_latent_bias_initializer,

                embedding_var_kernel_initializer=posterior_var_embedding_kernel_initializer,
                embedding_var_bias_initializer=posterior_var_embedding_bias_initializer,
                latent_var_kernel_initialiazer=posterior_var_latent_kernel_initialiazer,
                latent_var_bias_initializer=posterior_var_latent_bias_initializer,
            )
        with tf.name_scope('graph_px_g_y'):
            if self.kind == "binary":
                self.graphs_px_g_zy = SigmoidEncoder(
                    latent_dimension=self.in_dim, 
                    embedding_dimensions=self.em_dim[::-1], 
                    var_scope=self.v_name('graph_px_g_y'),
                    bn_before=self.bn_before,
                    bn_after=self.bn_after,
                    epsilon=self.rec_eps,

                    embedding_kernel_initializer=recon_embedding_kernel_initializer,
                    embedding_bias_initializer=recon_embedding_bias_initializer,
                    latent_kernel_initialiazer=recon_latent_kernel_initialiazer,
                    latent_bias_initializer=recon_latent_bias_initializer,

                )
            #else:
            #    self.graphs_px_g_zy = NormalDecoder(self.in_dim, self.em_dim[::-1])

    @tf.function#
    def call(self, x, y, training=False):

        xy = tf.concat([x, y], axis=-1)
        (
            qz_g_xy__sample,
            qz_g_xy__logprob,
            qz_g_xy__prob,
            qz_g_xy__mu,
            qz_g_xy__logvar
        ) = self.graphs_qz_g_xy.call(xy, training)
        (
            pz_g_y__sample,
            pz_g_y__logprob,
            pz_g_y__prob,
            pz_gy__mu,
            pz_gy__logvar,
        ) = self.graphs_pz_g_y.call(y,  training, qz_g_xy__sample)
        dkl_z_g_xy = self.graphs_pz_g_y.entropy(
            y, qz_g_xy__sample, qz_g_xy__mu, tf.exp(qz_g_xy__logvar),
            self.lat_eps
        )
        (
            px_g_zy__sample,
            px_g_zy__logprob,
            px_g_zy__prob,
        ) = self.graphs_px_g_zy.call(qz_g_xy__sample, training, x)

        return (
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
        )

    @tf.function(experimental_relax_shapes=True)
    def sample(self, samples, x, y, training=False):
        with tf.device("/gpu:0"):
            result = [self.call(x, y, training) for j in range(samples)]
            result_pivot = list(zip(*result))
        return result_pivot

    @staticmethod
    @tf.function
    def mc_stack_mean(x):
        return tf.reduce_sum(tf.stack(x, 0), 0) / len(x)

    @tf.function(experimental_relax_shapes=True)
    def monte_carlo_estimate(self, samples, x, y, training=False):
        return [
            self.mc_stack_mean(z)
            for z in self.sample(samples, x, y, training=False)
        ]



class Gmvae(Model, Scope):
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
        var_scope='gmvae',

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

        optimizer=tf.keras.optimizers.SGD(0.001)
    ):

        # instatiate
        Model.__init__(self)
        Scope.__init__(self, var_scope)

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

        if cat_latent_bias_initializer is None:
            cat_latent_bias_initializer = tf.initializers.constant(
                np.log((1/self.k)/(1-1/self.k))
            )
        # instantiate all variables in the graph

        self.z_kl_lambda = z_kl_lambda
        self.c_kl_lambda = c_kl_lambda
        # 
        with tf.name_scope('categorical'):
            self.graph_qy_g_x = CategoricalEncoder(
                latent_dimension=self.k, 
                embedding_dimensions=self.em_dim, 
                embedding_activation=self.em_act,
                var_scope=self.v_name('categorical_encoder'),
                bn_before=self.bn_before,
                bn_after=self.bn_after,
                epsilon=self.cat_eps,
                embedding_kernel_initializer=cat_embedding_kernel_initializer,
                embedding_bias_initializer=cat_embedding_bias_initializer,
                latent_kernel_initialiazer=cat_latent_kernel_initialiazer,
                latent_bias_initializer=cat_latent_bias_initializer
            )
            
            self.graph_qy_g_x_ohe = GumbleSoftmaxLayer()

        self.marginal_autoencoder = \
            MarginalAutoEncoder(
                self.in_dim, self.mem_dim, self.mem_lat, kind=self.kind,
                var_scope=self.v_name('marginal_autoencoder'),
                latent_epsilon=self.lat_eps,
                reconstruction_epsilon=self.rec_eps,
                embedding_activations=self.em_act,
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
            )

        #self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.optimizer = optimizer

    def increment_cooling(self):
        self.cooling_distance += 1

    @tf.function
    def sample_one(self, inputs, training=False, temperature=1.0):
        x = inputs

        qy_g_x__logit, qy_g_x__prob = self.graph_qy_g_x(x, training)
        qy_g_x_ohe = self.graph_qy_g_x_ohe(
            qy_g_x__logit, 
            temperature
        )

        (
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
        ) = self.marginal_autoencoder(
            x, qy_g_x_ohe, training
        )

        return (
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
        )

    @tf.function 
    def sample_one_even(self, inputs, training=False):
        x = inputs

        y_ = tf.cast(
            tf.fill(tf.stack([tf.shape(x)[0], self.k]), 0.0),
            dtype=x.dtype
        )

        qy_g_x__prob = tf.cast(
            tf.fill(tf.stack([tf.shape(x)[0], self.k]), 0.0),
            dtype=x.dtype
        )

        (
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
        ) = self.marginal_autoencoder(
            x, y_, training
        )

        return (
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
        )


    @tf.function(experimental_relax_shapes=True)
    def sample(self, samples, x, training=False, temperature=1.0):
        with tf.device("/gpu:0"):
            result = [self.sample_one(x, training, temperature) for j in range(samples)]
            result_pivot = list(zip(*result))
        return result_pivot

    @tf.function(experimental_relax_shapes=True)
    def sample_even(self, samples, x, training=False):
        with tf.device("/gpu:0"):
            result = [self.sample_one_even(x, training) for j in range(samples)]
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


    @tf.function(experimental_relax_shapes=True)
    def monte_carlo_estimate_even(self, samples, x, training=False):
        return [
            self.mc_stack_mean(z)
            for z in self.sample_even(samples, x, training=False, )
        ]


    @tf.function
    def call(self, x, training=False, samples=1, temperature=1.0):

        y_ = tf.cast(
            tf.fill(tf.stack([tf.shape(x)[0], self.k]), 0.0),
            dtype=x.dtype
        )
        py = tf.cast(tf.fill(tf.shape(y_), 1 / self.k, name="prob"), x.dtype)

        (
            mc_qy_g_x__prob,
            mc_qz_g_xy__sample,
            mc_qz_g_xy__logprob,
            mc_qz_g_xy__prob,
            mc_pz_g_y__sample,
            mc_pz_g_y__logprob,
            mc_pz_g_y__prob,
            mc_dkl_z_g_xy,
            mc_px_g_zy__sample,
            mc_px_g_zy__logprob,
            mc_px_g_zy__prob,
        ) = self.monte_carlo_estimate(samples, x, training, temperature)


        #recon = qy_g_x[:, i] * tf.cast(mc_px_g_zy__logprob


        return (
            py,
            mc_qy_g_x__prob,
            mc_qz_g_xy__sample,
            mc_qz_g_xy__logprob,
            mc_qz_g_xy__prob,
            mc_pz_g_y__sample,
            mc_pz_g_y__logprob,
            mc_pz_g_y__prob,
            mc_dkl_z_g_xy,
            mc_px_g_zy__sample,
            mc_px_g_zy__logprob,
            mc_px_g_zy__prob,
        )

    @tf.function 
    def call_even(self, x, training=False, samples=1):

        y_ = tf.cast(
            tf.fill(tf.stack([tf.shape(x)[0], self.k]), 0.0),
            dtype=x.dtype
        )
        py = tf.cast(tf.fill(tf.shape(y_), 1 / self.k, name="prob"), x.dtype)

        (
            mc_qy_g_x__prob,
            mc_qz_g_xy__sample,
            mc_qz_g_xy__logprob,
            mc_qz_g_xy__prob,
            mc_pz_g_y__sample,
            mc_pz_g_y__logprob,
            mc_pz_g_y__prob,
            mc_dkl_z_g_xy,
            mc_px_g_zy__sample,
            mc_px_g_zy__logprob,
            mc_px_g_zy__prob,
        ) = self.monte_carlo_estimate_even(samples, x, training, )


        #recon = qy_g_x[:, i] * tf.cast(mc_px_g_zy__logprob


        return (
            py,
            mc_qy_g_x__prob,
            mc_qz_g_xy__sample,
            mc_qz_g_xy__logprob,
            mc_qz_g_xy__prob,
            mc_pz_g_y__sample,
            mc_pz_g_y__logprob,
            mc_pz_g_y__prob,
            mc_dkl_z_g_xy,
            mc_px_g_zy__sample,
            mc_px_g_zy__logprob,
            mc_px_g_zy__prob,
        )

    @tf.function
    def entropy_fn(self, inputs, training=False, samples=1, temperature=1.0):
        (
            py,
            qy_g_x,
            mc_qz_g_xy__sample,
            mc_qz_g_xy__logprob,
            mc_qz_g_xy__prob,
            mc_pz_g_y__sample,
            mc_pz_g_y__logprob,
            mc_pz_g_y__prob,
            mc_dkl_z_g_xy,
            mc_px_g_zy__sample,
            mc_px_g_zy__logprob,
            mc_px_g_zy__prob,
        ) = self.call(inputs, training=training, samples=samples, temperature=temperature)

        # reconstruction
        recon = mc_px_g_zy__logprob

        # z_entropy
        #z_entropy = mc_pz_g_y__logprob - mc_qz_g_xy__logprob
        z_entropy = mc_dkl_z_g_xy 

        # y_entropy
        y_entropy = tf.reduce_sum(
            qy_g_x * (tf.math.log(py) - tf.math.log(qy_g_x)),
            axis=-1,
        )

        # elbo = recon + z_entropy + y_entropy
        return recon, z_entropy, y_entropy

    @tf.function#(autograph=False)
    def elbo(self, inputs, training=False, samples=1, temperature=1.0):
        recon, z_entropy, y_entropy = self.entropy_fn(inputs, training, samples, temperature)
        return recon + self.z_kl_lambda * z_entropy + self.c_kl_lambda * y_entropy

    @tf.function#(autograph=False)
    def loss_fn(self, inputs, training=False, samples=1, temperature=1.0):
        return -self.elbo(inputs, training, samples, temperature)
    
    @tf.function
    def even_mixture_loss(self, inputs, training=False, samples=1):
        (
            py,
            qy_g_x,
            mc_qz_g_xy__sample,
            mc_qz_g_xy__logprob,
            mc_qz_g_xy__prob,
            mc_pz_g_y__sample,
            mc_pz_g_y__logprob,
            mc_pz_g_y__prob,
            mc_dkl_z_g_xy,
            mc_px_g_zy__sample,
            mc_px_g_zy__logprob,
            mc_px_g_zy__prob,
        ) = self.call_even(inputs, training=training, samples=samples)

        # reconstruction
        recon = mc_px_g_zy__logprob
        # z_entropy
        z_entropy = mc_dkl_z_g_xy

        return -(recon + z_entropy)

    @tf.function#(autograph=False)
    def train_step(self, x, samples=1, tenorboard=False, batch=False, temperature=1.0):

        if tenorboard:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/gradient_tape/train'

            writer = tf.summary.create_file_writer(train_log_dir)


        # for x in dataset:
        # Tensorflow dataset is iterable in eager mode
        with tf.device("/gpu:0"):
            with tf.GradientTape() as tape:
                if batch:
                    loss = tf.reduce_mean(
                        self.loss_fn(x, True, samples, temperature)
                    )
                else:
                    loss = (
                        self.loss_fn(x, True, samples, temperature)
                    )
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

    #@tf.function
    def pretrain_step(self, x, samples=1, batch=False):
        # for x in dataset:
        # Tensorflow dataset is iterable in eager mode
        target_vars = [
            v for v in self.trainable_variables  
            if 'gmvae/marginal_autoencoder' in v.name
        ]
        with tf.device("/gpu:0"):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                #tape.watch(target_vars)

                if batch:
                    loss = tf.reduce_mean(self.even_mixture_loss(x, True, samples))
                else:
                    loss = (self.even_mixture_loss(x, True, samples))
                # Update ops for batch normalization
                # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # with tf.control_dependencies(update_ops):

                gradients = tape.gradient(loss, self.trainable_variables)
                # Clipping
                gradients = [
                    None
                    if gradient is None
                    #else tf.clip_by_value(
                    #    gradient, -self.gradient_clip, self.gradient_clip
                    #)
                    else tf.clip_by_norm(
                        gradient, self.gradient_clip
                    )
                    for gradient in gradients
                ]

                self.optimizer.apply_gradients(
                    zip(gradients, self.trainable_variables)
                )

    @tf.function#(autograph=False)
    def predict(self, x, training=False):
        qy_g_x__logit, qy_g_x__prob = self.graph_qy_g_x(x, training)
        return qy_g_x__prob


    @tf.function
    def cluster_loss(self, x, y, training=False):
        qy_g_x__logit, qy_g_x__prob = self.graph_qy_g_x(x, training)
        #nent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #    logits=qy_g_x__logit, 
        #    labels=y
        #), axis=-1)

        nent = (
            - tf.add_n(
                [
                    y[:,i] * (
                        tf.math.log(qy_g_x__prob[:,i])
                        - tf.math.log(y[:,i])
                    )
                    for i in range(self.k)
                ]
            )
        )

        return nent

    def pretrain_categories_step(self, x, y, samples=1):
        y = tf.clip_by_value(y, 0.05, 0.95)
        with tf.device("/gpu:0"):
            with tf.GradientTape() as tape:
                loss = tf.add(
                    tf.reduce_mean(self.cluster_loss(x, y, True)),
                    tf.reduce_mean(self.loss_fn(x, True, samples, temperature=0.6))
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
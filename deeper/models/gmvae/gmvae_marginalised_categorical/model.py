import tensorflow as tf
from tensorflow.python.eager import context
import numpy as np
import datetime

from deeper.ops.distance import kl_divergence
from deeper.layers.binary import SigmoidEncoder
from deeper.layers.categorical import CategoricalEncoder
from deeper.probability_layers.gumble_softmax import GumbleSoftmaxLayer
from deeper.probability_layers.normal import (
    RandomNormalEncoder,
    lognormal_kl
)
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
        latent_var_latent_bias_initializer=tf.initializers.zeros(),

        posterior_mu_embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        posterior_mu_embedding_bias_initializer=tf.initializers.zeros(),
        posterior_mu_latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        posterior_mu_latent_bias_initializer=tf.initializers.zeros(),

        posterior_var_embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        posterior_var_embedding_bias_initializer=tf.initializers.zeros(),
        posterior_var_latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        posterior_var_latent_bias_initializer=tf.initializers.zeros(),

        recon_embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        recon_embedding_bias_initializer=tf.initializers.zeros(),
        recon_latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        recon_latent_bias_initializer=tf.initializers.zeros(),

        connected_weights=True,

        latent_mu_embedding_dropout=0.0,
        latent_var_embedding_dropout=0.0,
        posterior_mu_dropout=0.0,
        posterior_var_dropout=0.0,
        recon_dropouut=0.0,
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
        self.connected_weights = connected_weights


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

                connected_weights = connected_weights,

                embedding_mu_dropout=latent_mu_embedding_dropout,
                embedding_var_dropout=latent_var_embedding_dropout,
            )
        with tf.name_scope('graph_pz_g_y'):
            self.graphs_pz_g_y = RandomNormalEncoder(
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

                connected_weights = connected_weights,

                embedding_mu_dropout=posterior_mu_dropout,
                embedding_var_dropout=posterior_var_dropout,
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

                    embedding_dropout=recon_dropouut,
                )
            else:
                self.graphs_px_g_zy = RandomNormalEncoder(
                    self.in_dim, 
                    self.em_dim[::-1],
                    bn_before=self.bn_before,
                    bn_after=self.bn_after,
                    embedding_mu_dropout=recon_dropouut,
                    embedding_var_dropout=recon_dropouut,
                    fixed_var=1.0,
                    epsilon=self.rec_eps,
                    embedding_mu_kernel_initializer=recon_embedding_kernel_initializer,
                    embedding_mu_bias_initializer=recon_embedding_bias_initializer,
                    latent_mu_kernel_initialiazer=recon_latent_kernel_initialiazer,
                    latent_mu_bias_initializer=recon_latent_bias_initializer,
                )

    @tf.function#
    def call(self, x, y, training=False):

        xy = tf.concat([x, y], axis=-1)
        (
            qz_g_xy__sample,
            qz_g_xy__logprob,
            qz_g_xy__prob,
            qz_g_xy__mu,
            qz_g_xy__logvar,
            qz_g_xy__var
        ) = self.graphs_qz_g_xy.call(xy, training)
        (
            pz_g_y__sample,
            pz_g_y__logprob,
            pz_g_y__prob,
            pz_gy__mu,
            pz_gy__logvar,
            pz_gy__var,
        ) = self.graphs_pz_g_y.call(y, training, qz_g_xy__sample)
        dkl_z_g_xy = pz_g_y__logprob - qz_g_xy__logprob
        (
            px_g_zy__sample,
            px_g_zy__logprob,
            px_g_zy__prob,
        ) = self.graphs_px_g_zy.call(qz_g_xy__sample, training, x)[0:3]

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
        latent_var_latent_bias_initializer=tf.initializers.zeros(),

        posterior_mu_embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        posterior_mu_embedding_bias_initializer=tf.initializers.zeros(),
        posterior_mu_latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        posterior_mu_latent_bias_initializer=tf.initializers.zeros(),

        posterior_var_embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        posterior_var_embedding_bias_initializer=tf.initializers.zeros(),
        posterior_var_latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        posterior_var_latent_bias_initializer=tf.initializers.zeros(),

        recon_embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        recon_embedding_bias_initializer=tf.initializers.zeros(),
        recon_latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        recon_latent_bias_initializer=tf.initializers.zeros(),

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
                latent_bias_initializer=cat_latent_bias_initializer,
                embedding_dropout=categorical_latent_embedding_dropout

            )
            

        self.marginal_autoencoder = \
            MarginalAutoEncoder(
                self.in_dim, self.mem_dim, self.mem_lat, kind=self.kind,
                var_scope=self.v_name('marginal_autoencoder'),
                latent_epsilon=self.lat_eps,
                reconstruction_epsilon=self.rec_eps,
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
            )

        input_dropout=tf.keras.layers.Dropout(0.2)

        #self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.optimizer = optimizer


    def increment_cooling(self):
        self.cooling_distance += 1


    @tf.function
    def sample_one(self, inputs, training=False):
        x = inputs

        #x = tf.cast(tf.greater(tf.cast(x,tf.float32), tf.random_uniform(tf.shape(x), 0, 1)), tf.float32)
        #Add random binarizer

        y_ = tf.cast(
            tf.fill(tf.stack([tf.shape(x)[0], self.k]), 0.0),
            x.dtype
        )
        py = tf.cast(tf.fill((tf.shape(x)[0], self.k,), 1 / self.k, name="prob"), x.dtype)
        qy_g_x__logit, qy_g_x__prob = self.graph_qy_g_x(x, training)

        qz_g_xy__sample = [None] * self.k
        qz_g_xy__logprob = [None] * self.k
        qz_g_xy__prob = [None] * self.k

        pz_g_y__sample = [None] * self.k
        pz_g_y__logprob = [None] * self.k
        pz_g_y__prob = [None] * self.k
        dkl_z_g_xy = [None] * self.k

        px_g_zy__sample = [None] * self.k
        px_g_zy__logprob = [None] * self.k
        px_g_zy__prob = [None] * self.k

        for i in range(self.k):
            with tf.name_scope('mixture_{}'.format(i)):
                y = tf.add(
                    y_,
                    tf.constant(
                        np.eye(self.k)[i],
                        dtype=x.dtype,
                        name="y_one_hot_{}".format(i),
                    ),
                    name="hot_at_{}".format(i),
                )

                (
                    qz_g_xy__sample[i],
                    qz_g_xy__logprob[i],
                    qz_g_xy__prob[i],
                    pz_g_y__sample[i],
                    pz_g_y__logprob[i],
                    pz_g_y__prob[i],
                    dkl_z_g_xy[i],
                    px_g_zy__sample[i],
                    px_g_zy__logprob[i],
                    px_g_zy__prob[i],
                ) = self.marginal_autoencoder(
                    x, y, training
                )

        # Losses
        # reconstruction
        recon = tf.add_n(
            [
                qy_g_x__prob[:, i] * (px_g_zy__logprob[i])
                for i in range(self.k)
            ]
        )

        # z_entropy
        z_entropy = tf.add_n(
            [
                qy_g_x__prob[:, i] * (dkl_z_g_xy[i])
                for i in range(self.k)
            ]
        )

        # y_entropy
        # E_q [log(p/q)] = sum q (log_p - log_q)
        y_entropy = (
            tf.reduce_sum(qy_g_x__prob * tf.math.log(py),-1)  
            + tf.nn.softmax_cross_entropy_with_logits(
                logits=qy_g_x__logit, 
                labels=qy_g_x__prob
            )
        )
        #tf.reduce_sum(
        #    qy_g_x__prob * (tf.math.log(py) - tf.math.log(qy_g_x__prob)),
        #    axis=-1,
        #)

        # elbo
        elbo = recon + self.z_kl_lambda * z_entropy + self.c_kl_lambda * y_entropy

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
            recon,
            z_entropy,
            y_entropy,
            elbo
        )

    @tf.function 
    def sample_one_even(self, inputs, training=False):
        x = inputs

        y_ = tf.cast(
            tf.fill(tf.stack([tf.shape(x)[0], self.k]), 0.0),
            x.dtype
        )

        qz_g_xy__sample = [None] * self.k
        qz_g_xy__logprob = [None] * self.k
        qz_g_xy__prob = [None] * self.k

        pz_g_y__sample = [None] * self.k
        pz_g_y__logprob = [None] * self.k
        pz_g_y__prob = [None] * self.k
        dkl_z_g_xy = [None] * self.k

        px_g_zy__sample = [None] * self.k
        px_g_zy__logprob = [None] * self.k
        px_g_zy__prob = [None] * self.k

        for i in range(self.k):
            with tf.name_scope('mixture_{}'.format(i)):
                y = tf.add(
                    y_,
                    tf.constant(
                        np.eye(self.k)[i],
                        dtype=x.dtype,
                        name="y_one_hot_{}".format(i),
                    ),
                    name="hot_at_{}".format(i),
                )

                (
                    qz_g_xy__sample[i],
                    qz_g_xy__logprob[i],
                    qz_g_xy__prob[i],
                    pz_g_y__sample[i],
                    pz_g_y__logprob[i],
                    pz_g_y__prob[i],
                    dkl_z_g_xy[i],
                    px_g_zy__sample[i],
                    px_g_zy__logprob[i],
                    px_g_zy__prob[i],
                ) = self.marginal_autoencoder(
                    x, y, training
                )

        # Losses
        # reconstruction
        recon = tf.add_n(
            [
                qy_g_x[:, i] * (px_g_zy__logprob[i])
                for i in range(self.k)
            ]
        )

        # z_entropy
        z_entropy = tf.add_n(
            [
                qy_g_x[:, i] * (pz_g_y__logprob[i] - qz_g_xy__logprob[i])
                for i in range(self.k)
            ]
        )

        # y_entropy
        y_entropy = 0.0

        # elbo
        elbo = recon + z_entropy + y_entropy


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
            recon,
            z_entropy,
            y_entropy,
            elbo
        )


    @tf.function(experimental_relax_shapes=True)
    def sample(self, samples, x, training=False):
        with tf.device("/gpu:0"):
            result = [self.sample_one(x, training) for j in range(samples)]
            result_pivot = list(zip(*result))
        return result_pivot


    @staticmethod
    @tf.function
    def mc_stack_mean(x):
        return tf.reduce_sum(tf.stack(x, 0), 0) / len(x)


    @tf.function(experimental_relax_shapes=True)
    def monte_carlo_estimate(self, samples, x, training=False):
        return [
            self.mc_stack_mean(z)
            for z in self.sample(samples, x, training=False)
        ]


    @tf.function
    def call(self, x, training=False, samples=1):


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
            recon,
            z_entropy,
            y_entropy,
            elbo
        ) = self.monte_carlo_estimate(samples, x, training)


        #recon = qy_g_x[:, i] * tf.cast(mc_px_g_zy__logprob


        return (
            py,                     #0
            mc_qy_g_x__prob,        #1
            mc_qz_g_xy__sample,     #2
            mc_qz_g_xy__logprob,    #3
            mc_qz_g_xy__prob,       #4
            mc_pz_g_y__sample,      #5
            mc_pz_g_y__logprob,     #6
            mc_pz_g_y__prob,        #7
            mc_dkl_z_g_xy,          #8
            mc_px_g_zy__sample,     #9
            mc_px_g_zy__logprob,    #10
            mc_px_g_zy__prob,       #11
            recon,                  #12
            z_entropy,              #13
            y_entropy,              #14
            elbo                    #15
        )

    @tf.function 
    def call_even(self, x, training=False, samples=1):

        y_ = tf.cast(
            tf.fill(tf.stack([tf.shape(x)[0], self.k]), 0.0),
            dtype=x.dtype
        )
        py = tf.cast(tf.fill(tf.shape(y_), 1/k, name="prob"), x.dtype)

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
            recon,
            z_entropy,
            y_entropy,
            elbo
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
            recon,
            z_entropy,
            y_entropy,
            elbo
        )


    @tf.function
    def latent_sample(self, inputs, training=False, samples=1 ):
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
            recon,
            z_entropy,
            y_entropy,
            elbo
        ) = self.call(inputs, training=training, samples=samples)

        latent = tf.add_n(
            [
                qy_g_x[:, i] * (mc_pz_g_y__sample[i])
                for i in range(self.k)
            ]
        )

        return latent


    @tf.function
    def entropy_fn(self, inputs, training=False, samples=1 ):
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
            recon,
            z_entropy,
            y_entropy,
            elbo
        ) = self.call(inputs, training=training, samples=samples)

        # elbo = recon + z_entropy + y_entropy
        return recon, z_entropy, y_entropy

    @tf.function#(autograph=False)
    def elbo(self, inputs, training=False, samples=1, beta_z=1.0, beta_y=1.0):
        recon, z_entropy, y_entropy = self.entropy_fn(inputs, training, samples)
        return recon + beta_z * z_entropy + beta_y * y_entropy


    @tf.function#(autograph=False)
    def loss_fn(self, inputs, training=False, samples=1, beta_z=1.0, beta_y=1.0):
        return -self.elbo(inputs, training, samples, beta_z, beta_y)
    

    @tf.function 
    def loss_fn_with_known_clusters(
        self, 
        x, 
        y,
        training=False, 
        samples=1, 
        beta_z=1.0, 
        beta_y=1.0
        ):

        even_elbo = self.elbo(x, training, samples, beta_z, 0.0)
        cluster_loss = self.cluster_loss(x, y, True)

        return - even_elbo + beta_y * cluster_loss

    @tf.function#(autograph=False)
    def train_step(
        self, 
        x, 
        samples=1, 
        tenorboard=False, 
        batch=False, 
        beta_z=1.0, 
        beta_y=1.0
        ):

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
                        self.loss_fn(x, True, samples, beta_z, beta_y)
                    )
                else:
                    loss = (
                        self.loss_fn(x, True, samples, beta_z, beta_y)
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
                if self.gradient_clip is not None
                else 
                gradient
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
    def pretrain_step(self, x, samples=1, batch=False, beta_z=1.0):
        self.train_step(self, x, samples, batch, beta_z, 0.0)


    @tf.function
    def predict(self, x, training=False):
        qy_g_x__logit, qy_g_x__prob = self.graph_qy_g_x(x, training)
        return qy_g_x__prob


    @tf.function
    def cluster_loss(self, x, y, training=False):
        qy_g_x__logit, qy_g_x__prob = self.graph_qy_g_x(x, training)
        nent = - tf.add_n(
            [
                y[:,i] * (tf.math.log(qy_g_x__prob[:,i]) - tf.math.log(y[:,i]))
                for i in range(self.k)
            ]
        )
        return nent

    def pretrain_categories_step(self, x, y, samples=1, beta_z=1.0, beta_y=1.0):
        #y = tf.clip_by_value(y, 0.05, 0.95)

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
                else 
                gradient
                for gradient in gradients
            ]

        #update
        with tf.device("/gpu:0"):
            self.optimizer.apply_gradients(
                zip(gradients, self.trainable_variables)
            )
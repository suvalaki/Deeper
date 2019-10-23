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

        latent_fixed_var=None,
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

                fixed_var=latent_fixed_var,
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

                fixed_var=latent_fixed_var,
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

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
from deeper.utils.sampling import mc_stack_mean_dict
from deeper.utils.function_helpers.decorators import inits_args
from deeper.utils.function_helpers.collectors import get_local_tensors
from deeper.utils.scope import Scope

tfk = tf.keras

Model = tfk.Model

class MarginalAutoEncoder(Model, Scope):

    @inits_args
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



        with tf.name_scope('graph_qz_g_xy'):
            self.graphs_qz_g_xy = RandomNormalEncoder(
                latent_dimension=self.latent_dim, 
                embedding_dimensions=self.embedding_dimensions, 
                var_scope=self.v_name('graph_qz_g_xy'),
                bn_before=self.bn_before,
                bn_after=self.bn_after,
                epsilon=self.latent_epsilon,

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
                latent_dimension=self.latent_dim, 
                embedding_dimensions=[], 
                var_scope=self.v_name('graph_pz_g_y'),
                bn_before=self.bn_before,
                bn_after=self.bn_after,
                epsilon=self.latent_prior_epsilon,

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
                    latent_dimension=self.input_dimension, 
                    embedding_dimensions=self.embedding_dimensions[::-1], 
                    var_scope=self.v_name('graph_px_g_y'),
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
                self.graphs_px_g_zy = RandomNormalEncoder(
                    self.input_dimension, 
                    self.embedding_dimensions[::-1],
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

    #@tf.function#
    def call(self, x, y, training=False):
        y = tf.cast(y, dtype=x.dtype)
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
            pz_g_y__mu,
            pz_g_y__logvar,
            pz_g_y__var,
        ) = self.graphs_pz_g_y.call(y, training, qz_g_xy__sample)
        dkl_z_g_xy = pz_g_y__logprob - qz_g_xy__logprob
        (
            px_g_zy__sample,
            px_g_zy__logprob,
            px_g_zy__prob,
        ) = self.graphs_px_g_zy.call(qz_g_xy__sample, training, x)[0:3]

        output = {
            'qz_g_xy__sample':qz_g_xy__sample,
            'qz_g_xy__logprob':qz_g_xy__logprob,
            'qz_g_xy__prob':qz_g_xy__prob,
            'qz_g_xy__mu':qz_g_xy__mu,
            'qz_g_xy__logvar':qz_g_xy__logvar,
            'qz_g_xy__var':qz_g_xy__var,
            'pz_g_y__sample':pz_g_y__sample,
            'pz_g_y__logprob':pz_g_y__logprob,
            'pz_g_y__prob':pz_g_y__prob,
            'pz_g_y__mu':pz_g_y__mu,
            'pz_g_y__logvar':pz_g_y__logvar,
            'pz_g_y__var':pz_g_y__var,
            'px_g_zy__sample':px_g_zy__sample,
            'px_g_zy__logprob':px_g_zy__logprob,
            'px_g_zy__prob':px_g_zy__prob,
        }

        return output


    @tf.function
    def sample(self, samples, x, y, training=False):
        with tf.device("/gpu:0"):
            result = [self.call(x, y, training) for j in range(samples)]
        return result


    @tf.function(experimental_relax_shapes=True)
    def monte_carlo_estimate(self, samples, x, y, training=False):
        return mc_stack_mean_dict(self.sample(samples, x, y, training))

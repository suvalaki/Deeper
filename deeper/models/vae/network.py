import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Activation

from typing import Tuple, Union, Optional, Sequence

from deeper.layers.encoder import Encoder
from deeper.probability_layers.normal import RandomNormalEncoder
from deeper.models.vae.utils import split_inputs
from deeper.utils.function_helpers.decorators import inits_args

from deeper.probability_layers.normal import (
    lognormal_kl,
    lognormal_pdf,
)


class VaeNet(Layer):

    output_names = [
        # Input variables
        "x_regression",
        "x_bin",
        "x_ord_groups_concat",
        "x_cat_groups_concat",
        # Encoder Variables
        "qz_g_x__sample",
        "qz_g_x__logprob",
        "qz_g_x__prob",
        "qz_g_x__mu",
        "qz_g_x__logvar",
        "qz_g_x__var",
        # DecoderVariables
        "x_recon",
        "x_recon_regression",
        "x_recon_bin_logit",
        "x_recon_bin",
        "x_recon_ord_groups_logit_concat",
        "x_recon_ord_groups_concat",
        "x_recon_cat_groups_logit_concat",
        "x_recon_cat_groups_concat",
    ]

    @inits_args
    def __init__(
        self,
        input_regression_dimension: int,
        input_boolean_dimension: int,
        input_ordinal_dimension: Union[int, Sequence[int]],
        input_categorical_dimension: Union[int, Sequence[int]],
        output_regression_dimension: int,
        output_boolean_dimension: int,
        output_ordinal_dimension: Union[int, Sequence[int]],
        output_categorical_dimension: Union[int, Sequence[int]],
        encoder_embedding_dimensions: Tuple[int],
        decoder_embedding_dimensions: Tuple[int],
        latent_dim: int,
        embedding_activations=tf.nn.relu,
        var_scope: str = "variational_autoencoder",
        bn_before: bool = False,
        bn_after: bool = False,
        latent_epsilon=0.0,
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
        connected_weights: bool = True,
        latent_mu_embedding_dropout: Optional[float] = None,
        latent_var_embedding_dropout: Optional[float] = None,
        recon_dropouut: Optional[float] = None,
        latent_fixed_var: Optional[float] = None,
        **kwargs,
    ):
        Layer.__init__(self, **kwargs)

        # Input Dimension Calculation
        (
            self.input_ordinal_dimension,
            self.input_ordinal_dimension_tot,
            self.input_categorical_dimension,
            self.input_categorical_dimension_tot,
            self.input_dim,
        ) = self.unpack_dimensions(
            input_regression_dimension,
            input_boolean_dimension,
            input_ordinal_dimension,
            input_categorical_dimension,
        )
        (
            self.output_ordinal_dimension,
            self.output_ordinal_dimension_tot,
            self.output_categorical_dimension,
            self.output_categorical_dimension_tot,
            self.output_dim,
        ) = self.unpack_dimensions(
            output_regression_dimension,
            output_boolean_dimension,
            output_ordinal_dimension,
            output_categorical_dimension,
        )

        # Encoder
        self.graph_qz_g_x = RandomNormalEncoder(
            latent_dimension=latent_dim,
            embedding_dimensions=encoder_embedding_dimensions,
            bn_before=bn_before,
            bn_after=bn_after,
            epsilon=latent_epsilon,
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
            decoder_embedding_dimensions,
            activation=embedding_activations,
            bn_before=bn_before,
            bn_after=bn_after,
            embedding_kernel_initializer=recon_embedding_kernel_initializer,
            embedding_bias_initializer=recon_embedding_bias_initializer,
            latent_kernel_initialiazer=recon_latent_kernel_initialiazer,
            latent_bias_initializer=recon_latent_bias_initializer,
            embedding_dropout=recon_dropouut,
        )

    @staticmethod
    def unpack_dimensions(
        reg_dim: int,
        bool_dim: int,
        ord_dim: Union[int, Sequence],
        cat_dim: Union[int, Sequence],
    ):
        ord_dim = ord_dim if isinstance(ord_dim, Sequence) else (ord_dim,)
        cat_dim = cat_dim if isinstance(cat_dim, Sequence) else (cat_dim,)
        tot_ord_dim = sum(ord_dim)
        tot_cat_dim = sum(cat_dim)
        tot_dim = reg_dim + bool_dim + tot_ord_dim + tot_cat_dim
        return (
            ord_dim,
            tot_ord_dim,
            cat_dim,
            tot_cat_dim,
            tot_dim,
        )

    @tf.function
    def split_inputs(self, x):
        return split_inputs(
            x,
            self.input_regression_dimension,
            self.input_boolean_dimension,
            self.input_ordinal_dimension,
            self.input_categorical_dimension,
        )

    @tf.function
    def split_outputs(self, x):
        return split_inputs(
            x,
            self.output_regression_dimension,
            self.output_boolean_dimension,
            self.output_categorical_dimension,
            self.output_categorical_dimension,
        )

    @tf.function
    def call(self, x, training=False):

        x = tf.cast(x, dtype=self.dtype)
        (
            x_regression,
            x_bin,
            x_ord_groups_concat,
            x_ord_groups,
            x_cat_groups_concat,
            x_cat_groups,
        ) = self.split_inputs(
            x,
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
            x_recon_ord_groups_logit_concat,
            x_recon_ord_groups_logit,
            x_recon_cat_groups_logit_concat,
            x_recon_cat_groups_logit,
        ) = self.split_outputs(
            out_hidden,
        )

        x_recon_bin = tf.nn.sigmoid(x_recon_bin_logit)
        x_recon_ord_groups = [
            tf.nn.sigmoid(x) for x in x_recon_ord_groups_logit
        ]
        x_recon_ord_groups_concat = (
            tf.nn.softmax(x_recon_ord_groups[0])
            if len(x_recon_ord_groups_logit) <= 1
            else tf.concat(
                [tf.nn.sigmoid(z) for z in x_recon_ord_groups_logit], -1
            )
        )
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
        result = [
            # Input variables
            x_regression,
            x_bin,
            x_ord_groups_concat,
            x_cat_groups_concat,
            # Encoder Variables
            qz_g_x__sample,
            qz_g_x__logprob,
            qz_g_x__prob,
            qz_g_x__mu,
            qz_g_x__logvar,
            qz_g_x__var,
            # DecoderVariables
            out_hidden,
            x_recon_regression,
            x_recon_bin_logit,
            x_recon_bin,
            x_recon_ord_groups_logit_concat,
            x_recon_ord_groups_concat,
            x_recon_cat_groups_logit_concat,
            x_recon_cat_groups_concat,
        ]

        return result

    @classmethod
    def call_to_dict(self, result):
        return {k: v for k, v in zip(self.output_names, result)}

    @tf.function
    def call_dict(self, x, training=False):
        return self.call_to_dict(self.call(x, training))

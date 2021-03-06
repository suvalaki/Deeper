import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Activation

from typing import Tuple, Union, Optional, Sequence

from deeper.layers.encoder import Encoder
from deeper.probability_layers.normal import (
    RandomNormalEncoder,
)
from deeper.probability_layers.ops.normal import std_normal_kl_divergence
from deeper.models.vae.utils import split_inputs
from deeper.utils.function_helpers.decorators import inits_args

from deeper.probability_layers.normal import (
    lognormal_kl,
    lognormal_pdf,
)
from tensorflow.python.keras.metrics import categorical_accuracy, accuracy


class VaeNet(Layer):

    output_names = [
        # Input variables
        # "x_regression",
        # "x_bin",
        # "x_ord_groups_concat",
        # "x_cat_groups_concat",
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
            # x_regression,
            # x_bin,
            # x_ord_groups_concat,
            # x_cat_groups_concat,
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


class VaeLossNet(tf.keras.layers.Layer):
    def __init__(
        self,
        latent_eps=0.0,
        encoder_name="zgy",
        decoder_name="xgz",
        prefix="",
    ):
        super(VaeLossNet, self).__init__()
        self.latent_eps = latent_eps
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.prefix = prefix

    def categorical_accuracy_grouped(
        self,
        y_cat_true: Sequence[tf.Tensor],
        y_cat_pred: Sequence[tf.Tensor],
        training: bool = False,
    ):

        cat_accs = [
            categorical_accuracy(yct, ypc)
            for (yct, ypc) in zip(y_cat_true, y_cat_pred)
        ]
        for i, acc in enumerate(cat_accs):
            self.add_metric(
                acc, name=f"{self.decoder_name}_cat_accuracy_group_{i}"
            )
        cat_acc = tf.reduce_mean(tf.stack(cat_accs))
        self.add_metric(cat_acc, name=f"{self.decoder_name}_cat_accuracy")
        return cat_acc

    def latent_kl(self, mu, logvar, training=False, name="latent_kl"):
        kl = std_normal_kl_divergence(
            mu, logvar, epsilon=self.latent_eps, name=name
        )
        self.add_metric(kl, name=name)
        return kl

    def xent_binary(
        self,
        y_bin_logits_true: Sequence[tf.Tensor],
        y_bin_logits_pred: Sequence[tf.Tensor],
        training: bool = False,
        weights: Optional[Sequence[float]] = None,
    ):
        if y_bin_logits_true.shape[1] > 0:
            xent = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    y_bin_logits_true,
                    y_bin_logits_pred,
                    name=f"{self.decoder_name}_binary_xent",
                ),
                -1,
            )
            self.add_metric(xent, name=f"{self.decoder_name}_binary_xent")
        else:
            xent = y_bin_logits_true[:, 0:0]
        return xent

    def xent_ordinal(
        self,
        y_ord_logits_true: Sequence[tf.Tensor],
        y_ord_logits_pred: Sequence[tf.Tensor],
        training: bool = False,
        class_weights: Optional[Sequence[float]] = None,
    ):

        if class_weights is not None:
            class_weights = [1 for i in range(y_ord_logits_true)]

        if len(y_ord_logits_true) == 0:
            return 0

        if len(y_ord_logits_true) == 1:
            xent = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    y_ord_logits_true,
                    y_ord_logits_pred,
                    name=f"{self.decoder_name}_ord_xent",
                ),
                -1,
            )
        else:
            xent = tf.math.add_n(
                [
                    wt
                    * tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            yolt,
                            yolp,
                            name=f"{self.decoder_name}_ord_xent_group_{i}",
                        ),
                        -1,
                    )
                    for i, (wt, yolt, yolp) in enumerate(
                        zip(
                            class_weights, y_ord_logits_true, y_ord_logits_pred
                        )
                    )
                ]
            )
        self.add_metric(xent, name=f"{self.decoder_name}_ord_xent")
        return xent

    def xent_categorical(
        self,
        y_ord_logits_true: Sequence[tf.Tensor],
        y_ord_logits_pred: Sequence[tf.Tensor],
        training: bool = False,
        class_weights: Optional[Sequence[float]] = None,
    ):
        if class_weights is not None:
            class_weights = [1 for i in range(y_ord_logits_true)]

        if len(y_ord_logits_pred) == 0:
            return 0

        if len(y_ord_logits_true) <= 1:
            xent = tf.nn.softmax_cross_entropy_with_logits(
                y_ord_logits_true,
                y_ord_logits_pred,
                name=f"{self.decoder_name}_cat_xent",
            )
        else:
            xents = [
                wt
                * tf.nn.softmax_cross_entropy_with_logits(
                    yolt,
                    yolp,
                    name=f"{self.decoder_name}_cat_xent_group_{i}",
                )
                for i, (wt, yolt, yolp) in enumerate(
                    zip(class_weights, y_ord_logits_true, y_ord_logits_pred)
                )
            ]
            xent = tf.add_n(xents)
            for i, x in enumerate(xents):
                self.add_metric(
                    x, name=f"{self.decoder_name}_cat_xent_group_{i}"
                )
        self.add_metric(xent, name=f"{self.decoder_name}_cat_xent")
        return xent

    def log_pxgz_regression(
        self,
        y_reg_true,
        y_reg_pred,
        training: bool = False,
    ):
        log_p = lognormal_pdf(y_reg_true, y_reg_pred, 1.0)
        self.add_metric(log_p, name=f"log_p{self.decoder_name}_regression")
        return log_p

    def log_pxgz_binary(
        self,
        y_bin_logits_true: Sequence[tf.Tensor],
        y_bin_logits_pred: Sequence[tf.Tensor],
        training: bool = False,
        weights: Optional[Sequence[float]] = None,
    ):
        if y_bin_logits_true.shape[0] > 0:
            log_p = -self.xent_binary(
                y_bin_logits_true, y_bin_logits_pred, training, weights
            )
            self.add_metric(log_p, name=f"log_p{self.decoder_name}_binary")
            return log_p
        else:
            return y_bin_logits_true[:, 0:0]

    def log_pxgz_ordinal(
        self,
        y_ord_logits_true: Sequence[tf.Tensor],
        y_ord_logits_pred: Sequence[tf.Tensor],
        training: bool = False,
        class_weights: Optional[Sequence[float]] = None,
    ):
        log_p = -self.xent_ordinal(
            y_ord_logits_true, y_ord_logits_pred, training, class_weights
        )
        self.add_metric(log_p, name=f"log_p{self.decoder_name}_ordinal")
        return log_p

    def log_pxgz_categorical(
        self,
        y_cat_logits_true: Sequence[tf.Tensor],
        y_cat_logits_pred: Sequence[tf.Tensor],
        training: bool = False,
        class_weights: Optional[Sequence[float]] = None,
    ):
        log_p = -self.xent_categorical(
            y_cat_logits_true, y_cat_logits_pred, training, class_weights
        )
        self.add_metric(log_p, name=f"log_p{self.decoder_name}_categorical")
        return log_p

    def log_pxgz(
        self,
        log_pxgz_reg,
        log_pxgz_bin,
        log_pxgz_ord,
        log_pxgz_cat,
        lambda_reg=1.0,
        lambda_bin=1.0,
        lambda_ord=1.0,
        lambda_cat=1.0,
    ):
        log_p = log_pxgz_reg + log_pxgz_bin + log_pxgz_ord + log_pxgz_cat
        self.add_metric(log_p, name=f"log_p{self.decoder_name}")

    def elbo(
        self,
        kl_z,
        ln_pxgz,
    ):
        result = ln_pxgz + kl_z
        self.add_metric(result, name=f"{self.prefix}elbo")
        return result

    def loss(
        self,
        kl_z,
        log_pxgz_reg,
        log_pxgz_bin,
        log_pxgz_ord,
        log_pxgz_cat,
        lambda_z=1.0,
        lambda_reg=1.0,
        lambda_bin=1.0,
        lambda_ord=1.0,
        lambda_cat=1.0,
    ):
        scaled_kl_z = lambda_z * kl_z
        self.add_metric(scaled_kl_z, name="scaled_latent_kl")
        scaled_log_pxgz_reg = lambda_reg * log_pxgz_reg
        self.add_metric(
            scaled_log_pxgz_reg, name=f"scaled_log_p{self.decoder_name}_reg"
        )
        scaled_log_pxgz_bin = lambda_bin * log_pxgz_bin
        self.add_metric(
            scaled_log_pxgz_bin, name=f"scaled_log_p{self.decoder_name}_bin"
        )
        scaled_log_pxgz_ord = lambda_ord * log_pxgz_ord
        self.add_metric(
            scaled_log_pxgz_ord, name=f"scaled_log_p{self.decoder_name}_ord"
        )
        scaled_log_pxgz_cat = lambda_reg * log_pxgz_cat
        self.add_metric(
            scaled_log_pxgz_cat, name=f"scaled_log_p{self.decoder_name}_cat"
        )
        scaled_log_pgz = (
            scaled_log_pxgz_reg
            + scaled_log_pxgz_bin
            + scaled_log_pxgz_ord
            + scaled_log_pxgz_cat
        )
        self.add_metric(
            scaled_log_pgz, name=f"scaled_log_p{self.encoder_name}"
        )
        scaled_elbo = scaled_log_pgz + scaled_kl_z
        self.add_metric(scaled_elbo, name=f"{self.prefix}scaled_elbo")
        scaled_loss = -scaled_elbo
        self.add_metric(scaled_loss, name=f"{self.prefix}scaled_loss")
        return scaled_loss

    def call(self, inputs, training=False):

        (
            (z_mu, z_logvar),
            (
                y_true_reg,
                y_true_bin,
                y_true_ord,
                y_true_cat,
            ),
            (
                y_pred_reg,
                y_pred_bin_logit,
                y_pred_ord_logit,
                y_pred_cat_logit,
            ),
            (
                lambda_z,
                lambda_reg,
                lambda_bin,
                lambda_ord,
                lambda_cat,
            ),
        ) = inputs

        kl_z = self.latent_kl(z_mu, z_logvar, training)
        l_pxgz_reg = self.log_pxgz_regression(y_true_reg, y_pred_reg, training)
        l_pxgz_bin = self.log_pxgz_binary(
            y_true_bin, y_pred_bin_logit, training
        )
        l_pxgz_ord = self.log_pxgz_ordinal(
            y_true_ord, y_pred_ord_logit, training
        )
        l_pxgz_cat = self.log_pxgz_categorical(
            y_true_cat, y_pred_cat_logit, training
        )
        loss = self.loss(
            kl_z,
            l_pxgz_reg,
            l_pxgz_bin,
            l_pxgz_ord,
            l_pxgz_cat,
            lambda_z,
            lambda_reg,
            lambda_bin,
            lambda_ord,
            lambda_cat,
        )
        return loss

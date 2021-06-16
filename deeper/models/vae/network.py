from __future__ import annotations
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
from deeper.models.vae.utils import split_inputs, SplitCovariates
from deeper.utils.function_helpers.decorators import inits_args

from deeper.probability_layers.normal import (
    lognormal_kl,
    lognormal_pdf,
)
from tensorflow.python.keras.metrics import categorical_accuracy, accuracy

from deeper.utils.tf.keras.models import Model
from typing import Sequence

from typing import NamedTuple
import numpy as np


def reduce_groups(fn, x_grouped: Sequence[tf.Tensor]):
    if len(x_grouped) <= 1:
        return fn(x_grouped[0])
    return tf.concat([fn(z) for z in x_grouped], -1)


class VaeNet(Layer):
    class ReconstructionOutput(NamedTuple):
        hidden_logits: tf.Tensor
        regression: tf.Tensor
        logits_binary: tf.Tensor
        binary: tf.Tensor
        logits_ordinal_groups_concat: tf.Tensor
        logits_ordinal_groups: tf.Tensor
        ord_groups_concat: tf.Tensor
        ord_groups: tf.Tensor
        logits_categorical_groups_concat: tf.Tensor
        logits_categorical_groups: tf.Tensor
        categorical_groups_concat: tf.Tensor
        categorical_groups: tf.Tensor

    class VaeNetOutput(NamedTuple):
        # Encoder/Latent variables
        qz_g_x__sample: tf.Tensor
        qz_g_x__logprob: tf.Tensor
        qz_g_x__prob: tf.Tensor
        qz_g_x__mu: tf.Tensor
        qz_g_x__logvar: tf.Tensor
        qz_g_x__var: tf.Tensor
        # DecoderVariables
        x_recon: tf.Tensor
        x_recon_regression: tf.Tensor
        x_recon_bin_logit: tf.Tensor
        x_recon_bin: tf.Tensor
        x_recon_ord_groups_logit_concat: tf.Tensor
        x_recon_ord_groups_logit: tf.Tensor
        x_recon_ord_groups_concat: tf.Tensor
        x_recon_ord_groups: tf.Tensor
        x_recon_cat_groups_logit_concat: tf.Tensor
        x_recon_cat_groups_logit: tf.Tensor
        x_recon_cat_groups_concat: tf.Tensor
        x_recon_cat_groups: tf.Tensor

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
    def split_inputs(self, x) -> SplitCovariates:
        return split_inputs(
            x,
            self.input_regression_dimension,
            self.input_boolean_dimension,
            self.input_ordinal_dimension,
            self.input_categorical_dimension,
        )

    @tf.function
    def split_outputs(self, x) -> SplitCovariates:
        return split_inputs(
            x,
            self.output_regression_dimension,
            self.output_boolean_dimension,
            self.output_categorical_dimension,
            self.output_categorical_dimension,
        )

    @tf.function
    def logits_to_actuals(
        self, output_logits_concat, training=False
    ) -> self.ReconstructionOutput:
        """Binary and categorical logits need to be converted into probs"""

        x_recon_logit = self.split_outputs(
            output_logits_concat,
        )

        x_recon_bin = tf.nn.sigmoid(x_recon_logit.binary)
        x_recon_ord_groups = [
            tf.nn.sigmoid(x) for x in x_recon_logit.ordinal_groups
        ]
        x_recon_ord_groups_concat = reduce_groups(
            tf.nn.softmax, x_recon_ord_groups
        )
        x_recon_cat_groups = [
            tf.nn.softmax(x) for x in x_recon_logit.categorical_groups
        ]
        x_recon_cat_groups_concat = reduce_groups(
            tf.nn.softmax, x_recon_cat_groups
        )

        return VaeNet.ReconstructionOutput(
            output_logits_concat,
            x_recon_logit.regression,
            x_recon_logit.binary,
            x_recon_bin,
            x_recon_logit.ordinal_groups_concat,
            x_recon_logit.ordinal_groups,
            x_recon_ord_groups_concat,
            x_recon_ord_groups,
            x_recon_logit.categorical_groups_concat,
            x_recon_logit.categorical_groups,
            x_recon_cat_groups_concat,
            x_recon_cat_groups,
        )

    @tf.function
    def call(self, x, training=False) -> VaeNet.VaeNetOutput:

        x = tf.cast(x, dtype=self.dtype)

        # Encoder
        graph_z_g_x = self.graph_qz_g_x.call(x, training)

        # Decoder
        graph_x_g_z = self.logits_to_actuals(
            self.graph_px_g_z.call(graph_z_g_x.sample, training), training
        )

        return VaeNet.VaeNetOutput(*graph_z_g_x, *graph_x_g_z)

    @classmethod
    def call_to_dict(self, result):
        return result._asdict()

    @tf.function
    def call_dict(self, x, training=False):
        return self.call_to_dict(self.call(x, training))


class VaeReconLossNet(tf.keras.layers.Layer):
    class InputYTrue(NamedTuple):
        regression_value: tf.Tensor
        binary_prob: tf.Tensor
        ordinal_prob: tf.Tensor
        categorical_prob: tf.Tensor

    class InputYPred(NamedTuple):
        regression_value: tf.Tensor
        binary_logit: tf.Tensor
        ordinal_logit: tf.Tensor
        categorical_logit: tf.Tensor

    class Input(NamedTuple):
        y_true: VaeReconLossNet.InputYTrue
        y_pred: VaeReconLossNet.InputYPred

    class Output(NamedTuple):
        l_pxgz_reg: tf.Tensor
        l_pxgz_bin: tf.Tensor
        l_pxgz_ord: tf.Tensor
        l_pxgz_cat: tf.Tensor

    def __init__(
        self,
        decoder_name="xgz",
        prefix="",
    ):
        super(VaeReconLossNet, self).__init__()
        self.decoder_name = decoder_name
        self.prefix = prefix

    @tf.function
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
                acc,
                name=f"{self.prefix}/{self.decoder_name}_cat_accuracy_group_{i}",
            )
        cat_acc = tf.reduce_mean(tf.stack(cat_accs))
        self.add_metric(
            cat_acc, name=f"{self.prefix}/{self.decoder_name}_cat_accuracy"
        )
        return cat_acc

    @tf.function
    def xent_binary(
        self,
        y_bin_logits_true: Sequence[tf.Tensor],
        y_bin_logits_pred: Sequence[tf.Tensor],
        training: bool = False,
        weights: Optional[Sequence[float]] = None,
    ):
        if y_bin_logits_true.shape[-1] > 0:
            xent = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    y_bin_logits_true,
                    y_bin_logits_pred,
                    name=f"{self.prefix}/{self.decoder_name}_binary_xent",
                ),
                -1,
            )
            self.add_metric(
                xent, name=f"{self.prefix}/{self.decoder_name}_binary_xent"
            )
        else:
            xent = y_bin_logits_true[:, 0:0]
        return xent

    @tf.function
    def xent_ordinal(
        self,
        y_ord_logits_true: Sequence[tf.Tensor],
        y_ord_logits_pred: Sequence[tf.Tensor],
        training: bool = False,
        class_weights: Optional[Sequence[float]] = None,
    ):

        xent = tf.zeros((tf.shape(y_ord_logits_pred)[0],), dtype=self.dtype)

        if class_weights is None and len(y_ord_logits_true) > 0:
            class_weights = [1 for i in range(len(y_ord_logits_true))]

        if len(y_ord_logits_true) == 0:
            return 0.0

        elif len(y_ord_logits_true) == 1:
            if y_ord_logits_pred[0].get_shape()[-1] == 1:
                xent = tf.zeros(
                    (tf.shape(y_ord_logits_pred)[0],), dtype=self.dtype
                )
            else:
                xent = tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        y_ord_logits_true[0],
                        y_ord_logits_pred[0],
                        name=f"{self.prefix}/{self.decoder_name}_ord_xent",
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
                            name=f"{self.prefix}/{self.decoder_name}_ord_xent_group_{i}",
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

        self.add_metric(
            xent, name=f"{self.prefix}/{self.decoder_name}_ord_xent"
        )
        return xent

    @tf.function
    def xent_categorical(
        self,
        y_ord_logits_true: Sequence[tf.Tensor],
        y_ord_logits_pred: Sequence[tf.Tensor],
        training: bool = False,
        class_weights: Optional[Sequence[float]] = None,
    ):
        xent = 0.0
        # logit = np.log(1e-4 / (1 - 1e-4))

        if class_weights is not None:
            class_weights = [1 for i in range(y_ord_logits_true)]

        if len(y_ord_logits_pred) == 0:
            return 0.0

        elif len(y_ord_logits_true) == 1:
            if y_ord_logits_true[-1].get_shape()[-1] <= 1:
                self.add_metric(
                    0, name=f"{self.prefix}/{self.decoder_name}_cat_xent"
                )
                return 0.0
            else:
                xent = tf.nn.softmax_cross_entropy_with_logits(
                    y_ord_logits_true[0],
                    y_ord_logits_pred[0],
                    name=f"{self.prefix}/{self.decoder_name}_cat_xent",
                )

                self.add_metric(
                    xent, name=f"{self.prefix}/{self.decoder_name}_cat_xent"
                )
                return xent
        else:
            xents = [
                wt
                * tf.nn.softmax_cross_entropy_with_logits(
                    yolt,
                    yolp,
                    name=f"{self.prefix}/{self.decoder_name}_cat_xent_group_{i}",
                )
                for i, (wt, yolt, yolp) in enumerate(
                    zip(class_weights, y_ord_logits_true, y_ord_logits_pred)
                )
            ]
            xent = tf.add_n(xents)
            for i, x in enumerate(xents):
                self.add_metric(
                    x,
                    name=f"{self.prefix}/{self.decoder_name}_cat_xent_group_{i}",
                )
        self.add_metric(
            xent, name=f"{self.prefix}/{self.decoder_name}_cat_xent"
        )
        return xent

    @tf.function
    def log_pxgz_regression(
        self,
        y_reg_true,
        y_reg_pred,
        training: bool = False,
    ):
        log_p = lognormal_pdf(y_reg_true, y_reg_pred, 1.0)
        self.add_metric(
            log_p, name=f"{self.prefix}/log_p{self.decoder_name}_regression"
        )
        return log_p

    @tf.function
    def log_pxgz_binary(
        self,
        y_bin_logits_true: Sequence[tf.Tensor],
        y_bin_logits_pred: Sequence[tf.Tensor],
        training: bool = False,
        weights: Optional[Sequence[float]] = None,
    ):
        if y_bin_logits_true.shape[1] > 0:
            log_p = -self.xent_binary(
                y_bin_logits_true, y_bin_logits_pred, training, weights
            )
            self.add_metric(
                log_p, name=f"{self.prefix}/log_p{self.decoder_name}_binary"
            )
            return log_p
        else:
            return tf.reduce_sum(y_bin_logits_true[:, 0:0], -1)

    @tf.function
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
        self.add_metric(
            log_p, name=f"{self.prefix}/log_p{self.decoder_name}_ordinal"
        )
        return log_p

    @tf.function
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
        self.add_metric(
            log_p, name=f"{self.prefix}/log_p{self.decoder_name}_categorical"
        )
        return log_p

    def call(
        self, x: VaeReconLossNet.Input, training=False
    ) -> VaeReconLossNet.Output:
        l_pxgz_reg = self.log_pxgz_regression(
            x.y_true.regression_value,
            x.y_pred.regression_value,
            training,
        )
        l_pxgz_bin = self.log_pxgz_binary(
            x.y_true.binary_prob, x.y_pred.binary_logit, training
        )
        l_pxgz_ord = self.log_pxgz_ordinal(
            x.y_true.ordinal_prob, x.y_pred.ordinal_logit, training
        )
        l_pxgz_cat = self.log_pxgz_categorical(
            x.y_true.categorical_prob,
            x.y_pred.categorical_logit,
            training,
        )
        return VaeReconLossNet.Output(
            l_pxgz_reg, l_pxgz_bin, l_pxgz_ord, l_pxgz_cat
        )


class VaeLossNetLatent(tf.keras.layers.Layer):
    class Input(NamedTuple):
        mu: tf.Tensor
        logvar: tf.Tensor

    def __init__(self, latent_eps=0.0, name="latent_kl", **kwargs):
        super(VaeLossNetLatent, self).__init__(name=name, **kwargs)
        self.latent_eps = latent_eps

    @tf.function
    def latent_kl(self, mu, logvar, training=False):
        kl = std_normal_kl_divergence(mu, logvar, epsilon=self.latent_eps)
        self.add_metric(kl, name=self.name)
        return kl

    @tf.function
    def call(self, x: VaeLossNetLatent.Input, training=False) -> tf.Tensor:
        return self.latent_kl(x.mu, x.logvar, training)


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
        self.latent_lossnet = VaeLossNetLatent(latent_eps, name="latent_kl")
        self.recon_lossnet = VaeReconLossNet(decoder_name, prefix)

    @tf.function
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

    @tf.function
    def elbo(
        self,
        kl_z,
        ln_pxgz,
    ):
        result = ln_pxgz + kl_z
        self.add_metric(result, name=f"{self.prefix}elbo")
        return result

    class Output(NamedTuple):
        kl_z: tf.Tensor
        l_pxgz_reg: tf.Tensor
        l_pxgz_bin: tf.Tensor
        l_pxgz_ord: tf.Tensor
        l_pxgz_cat: tf.Tensor
        scaled_elbo: tf.Tensor
        loss: tf.Tensor
        lambda_z: tf.Tensor
        lambda_reg: tf.Tensor
        lambda_bin: tf.Tensor
        lambda_ord: tf.Tensor
        lambda_cat: tf.Tensor

    @tf.function
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

        return self.Output(
            kl_z,
            log_pxgz_reg,
            log_pxgz_bin,
            log_pxgz_ord,
            log_pxgz_cat,
            scaled_elbo,
            scaled_loss,
            lambda_z,
            lambda_reg,
            lambda_bin,
            lambda_ord,
            lambda_cat,
        )

    class InputWeight(NamedTuple):
        lambda_z: float
        lambda_reg: float
        lambda_bin: float
        lambda_ord: float
        lambda_cat: float

    class Input(NamedTuple):
        latent: VaeLossNetLatent.Input
        y_true: VaeReconLossNet.InputYTrue
        y_pred: VaeReconLossNet.InputYPred
        weight: VaeLossNet.InputWeight

        @staticmethod
        def from_nested_sequence(inputs) -> VaeLossNet.Input:
            return VaeLossNet.Input(
                VaeLossNetLatent.Input(*inputs[0]),
                VaeReconLossNet.InputYTrue(*inputs[1]),
                VaeReconLossNet.InputYPred(*inputs[2]),
                VaeLossNet.InputWeight(*inputs[3]),
            )

        @staticmethod
        def from_vaenet_outputs(
            y_true: SplitCovariates,
            model_output: VaeNet.VaeNetOutput,
            weights: VaeLossNet.InputWeight,
        ) -> VaeLossNet.Input:
            return VaeLossNet.Input(
                VaeLossNetLatent.Input(
                    model_output.qz_g_x__mu, model_output.qz_g_x__logvar
                ),
                VaeReconLossNet.InputYTrue(
                    y_true.regression,
                    y_true.binary,
                    y_true.ordinal_groups,
                    y_true.categorical_groups,
                ),
                VaeReconLossNet.InputYPred(
                    model_output.x_recon_regression,
                    model_output.x_recon_bin_logit,
                    model_output.x_recon_ord_groups_logit,
                    model_output.x_recon_cat_groups_logit,
                ),
                weights,
            )

    @tf.function
    def call(self, inputs: Input, training=False) -> Output:

        if not isinstance(inputs, VaeLossNet.Input):
            inputs = VaeLossNet.Input.from_nested_sequence(inputs)

        kl_z = self.latent_lossnet(inputs.latent, training)
        (l_pxgz_reg, l_pxgz_bin, l_pxgz_ord, l_pxgz_cat) = self.recon_lossnet(
            VaeReconLossNet.Input(inputs.y_true, inputs.y_pred), training
        )

        return self.loss(
            kl_z,
            l_pxgz_reg,
            l_pxgz_bin,
            l_pxgz_ord,
            l_pxgz_cat,
            *inputs.weight,
        )

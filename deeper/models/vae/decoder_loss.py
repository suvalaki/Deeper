from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Activation

from typing import Tuple, Union, Optional, Sequence, NamedTuple
from dataclasses import dataclass

from deeper.layers.encoder import Encoder
from deeper.utils.function_helpers.decorators import inits_args
from deeper.layers.data_splitter import split_inputs, unpack_dimensions
from deeper.models.vae.encoder import VaeEncoderNet
from deeper.layers.data_splitter import DataSplitter, reduce_groups

from deeper.probability_layers.normal import (
    lognormal_kl,
    lognormal_pdf,
)
from tensorflow.python.keras.metrics import categorical_accuracy, accuracy


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

        @classmethod
        def from_VaeReconstructionNet(cls, x: VaeReconstructionNet.Output):
            return cls(
                x.regression,
                x.logits_binary,
                x.ord_groups,
                x.categorical_groups,
            )

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
        prefix="",  # Must be non null for freestanding scope. Cannot start with "/"
    ):
        super().__init__(self)
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
                    name=f"{self.prefix}/xent/{self.decoder_name}_binary_xent",
                ),
                -1,
            )
            self.add_metric(
                xent,
                name=f"{self.prefix}/xent/{self.decoder_name}_binary_xent",
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
                        name=f"{self.prefix}/xent/{self.decoder_name}_ord_xent",
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
                            name=f"{self.prefix}/xent/{self.decoder_name}_ord_xent_group_{i}",
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
            xent, name=f"{self.prefix}/xent/{self.decoder_name}_ord_xent"
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

        if class_weights is None:
            class_weights = [1 for i in range(len(y_ord_logits_true))]

        if len(y_ord_logits_pred) == 0:
            return 0.0

        elif len(y_ord_logits_true) == 1:
            if y_ord_logits_true[-1].get_shape()[-1] <= 1:
                self.add_metric(
                    0, name=f"{self.prefix}/xent/{self.decoder_name}_cat_xent"
                )
                return 0.0
            else:
                xent = tf.nn.softmax_cross_entropy_with_logits(
                    y_ord_logits_true[0],
                    y_ord_logits_pred[0],
                    name=f"{self.prefix}/xent/{self.decoder_name}_cat_xent",
                )

                self.add_metric(
                    xent,
                    name=f"{self.prefix}/xent/{self.decoder_name}_cat_xent",
                )
                return xent
        else:
            xents = [
                wt
                * tf.nn.softmax_cross_entropy_with_logits(
                    yolt,
                    yolp,
                    name=f"{self.prefix}/xent/{self.decoder_name}_cat_xent_group_{i}",
                    axis=-1,
                )
                for i, (wt, yolt, yolp) in enumerate(
                    zip(class_weights, y_ord_logits_true, y_ord_logits_pred)
                )
                if yolt.shape[-1] > 0
            ]
            xent = tf.math.add_n(xents) if len(xents) > 0 else 0.0
            for i, x in enumerate(xents):
                self.add_metric(
                    x,
                    name=f"{self.prefix}/xent/{self.decoder_name}_cat_xent_group_{i}",
                )
        self.add_metric(
            xent, name=f"{self.prefix}/xent/{self.decoder_name}_cat_xent"
        )
        return xent

    @tf.function
    def log_pxgz_regression(
        self,
        y_reg_true,
        y_reg_pred,
        training: bool = False,
    ):

        if y_reg_true.shape[-1] > 0:
            log_p = lognormal_pdf(y_reg_true, y_reg_pred, 1.0)
        else:
            log_p = y_reg_true[:, 0:0]
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
                log_p, name=f"{self.prefix}/log_p/{self.decoder_name}_binary"
            )
            # accuracy
            y_bin_pred = tf.cast(
                tf.nn.sigmoid(y_bin_logits_pred) > 0.5, dtype=tf.float32
            )
            acc = tf.reduce_mean(
                tf.cast(y_bin_logits_true == y_bin_pred, dtype=tf.float32)
            )
            self.add_metric(
                acc,
                name=f"{self.prefix}/log_p/{self.decoder_name}_binary_accuracy",
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
            log_p, name=f"{self.prefix}/log_p/{self.decoder_name}_ordinal"
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
            log_p, name=f"{self.prefix}/log_p/{self.decoder_name}_categorical"
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
        return self.Output(
            l_pxgz_reg, l_pxgz_bin, l_pxgz_ord, l_pxgz_cat
        )

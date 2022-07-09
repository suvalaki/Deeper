from __future__ import annotations

import tensorflow as tf
from deeper.models.autoencoder.network import AutoencoderNet
from deeper.models.autoencoder.utils import AutoencoderTypeGetter
from deeper.models.vae.decoder_loss import VaeReconLossNet
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Activation, Layer
from deeper.utils.tf.experimental.extension_type import ExtensionTypeIterableMixin


class AutoencoderLossNet(VaeReconLossNet, AutoencoderTypeGetter):
    class InputWeight(
        tf.experimental.ExtensionType, ExtensionTypeIterableMixin, AutoencoderTypeGetter
    ):
        lambda_reg: tf.Tensor = tf.constant(1.0)
        lambda_bin: tf.Tensor = tf.constant(1.0)
        lambda_ord: tf.Tensor = tf.constant(1.0)
        lambda_cat: tf.Tensor = tf.constant(1.0)

    class Input(tf.experimental.ExtensionType, ExtensionTypeIterableMixin, AutoencoderTypeGetter):
        y_true: VaeReconLossNet.InputYTrue
        y_pred: VaeReconLossNet.InputYPred
        weight: AutoencoderLossNet.InputWeight

        @staticmethod
        @tf.function
        def from_output(
            y_true: SplitCovariates,
            model_output: AutoencoderNet.VaeNetOutput,
            weights: AutoencoderLossNet.InputWeight,
        ) -> AutoencoderLossNet.Input:
            return AutoencoderLossNet.Input(
                VaeReconLossNet.InputYTrue(
                    y_true.regression,
                    y_true.binary,
                    y_true.ordinal_groups,
                    y_true.categorical_groups,
                ),
                VaeReconLossNet.InputYPred.from_VaeReconstructionNet(model_output.reconstruction),
                weights,
            )

    class Output(tf.experimental.ExtensionType, ExtensionTypeIterableMixin):
        losses: VaeReconLossNet.Output
        scaled: VaeReconLossNet.Output
        loss: tf.Tensor

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self, inputs: AutoencoderNet.Input, training=False, **kwargs):
        lins = VaeReconLossNet.Input(inputs.y_true, inputs.y_pred)
        # losses = super(VaeReconLossNet, self).call(lins, training=training)

        x = lins
        l_pxgz_reg = self.log_pxgz_regression(
            x.y_true.regression_value,
            x.y_pred.regression_value,
            training,
        )
        l_pxgz_bin = self.log_pxgz_binary(x.y_true.binary_prob, x.y_pred.binary_logit, training)
        l_pxgz_ord = self.log_pxgz_ordinal(x.y_true.ordinal_prob, x.y_pred.ordinal_logit, training)
        l_pxgz_cat = self.log_pxgz_categorical(
            x.y_true.categorical_prob,
            x.y_pred.categorical_logit,
            training,
        )

        losses = VaeReconLossNet.Output(l_pxgz_reg, l_pxgz_bin, l_pxgz_ord, l_pxgz_cat)
        scaled = VaeReconLossNet.Output(
            l_pxgz_reg=inputs.weight.lambda_reg * losses.l_pxgz_reg,
            l_pxgz_bin=inputs.weight.lambda_bin * losses.l_pxgz_bin,
            l_pxgz_ord=inputs.weight.lambda_ord * losses.l_pxgz_ord,
            l_pxgz_cat=inputs.weight.lambda_cat * losses.l_pxgz_cat,
        )

        return AutoencoderLossNet.Output(losses, scaled, -sum(scaled))

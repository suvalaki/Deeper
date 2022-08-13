from __future__ import annotations

import tensorflow as tf
from typing import NamedTuple, Union

from tensorflow.python.util.tf_export import keras_export

from deeper.models.gan.network import (
    GanNet,
    GanGenerativeNet,
    GanDescriminativeNet,
)


def coalsece_nm(x, y):
    if x is not None:
        return x + "/" + y
    return y


class GanGeneratorLossNet(tf.keras.layers.Layer):

    # Reconstruction + get gan to fool descrim
    # Measures the ability for the generator to fool the descriminator. Samples should fool the desciminator
    def __init__(self, prefix=None, **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix

    def call(
        self,
        y_true: Union[tf.Tensor, NamedTuple, tf.experimental.ExtensionType],
        y_pred: GanGenerativeNet.Output,
        training: bool = False,
    ):
        descrim_t = tf.ones_like(y_pred.fake_descriminant.logits)
        descrim_tuning_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=descrim_t, logits=y_pred.fake_descriminant.logits
        )

        # accuracy
        y_bin_pred_real = tf.cast(
            tf.nn.sigmoid(y_pred.fake_descriminant.logits) > 0.5,
            dtype=tf.float32,
        )
        y_bin_true_real = tf.cast(descrim_t > 0.5, dtype=tf.float32)
        acc = tf.reduce_mean(tf.cast(y_bin_pred_real == y_bin_true_real, dtype=tf.float32))

        self.add_metric(
            acc, aggregation="mean", name=coalsece_nm(self.prefix, "generator_accuracy")
        )

        return descrim_tuning_loss


class GanDescriminationLossNet(tf.keras.layers.Layer):

    # make gan wise to being fooled
    # Reconstruction + get gan to fool descrim
    # Measure the descriminators ability to distinguish between real and fake
    def __init__(self, prefix=None, **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix

    def call(
        self,
        y_true: Union[tf.Tensor, NamedTuple, tf.experimental.ExtensionType],
        y_pred: GanDescriminativeNet.Output,
        training: bool = False,
    ):
        descrim_f = tf.zeros_like(y_pred.fake_descriminant.logits)
        descrim_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=descrim_f, logits=y_pred.fake_descriminant.logits
        )

        descrim_t = tf.ones_like(y_pred.real_descriminant.logits)
        descrim_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=descrim_t, logits=y_pred.real_descriminant.logits
        )

        # accuracy
        y_bin_pred_fake = tf.cast(
            tf.nn.sigmoid(y_pred.fake_descriminant.logits) > 0.5,
            dtype=tf.float32,
        )
        y_bin_true_fake = tf.cast(descrim_f > 0.5, dtype=tf.float32)
        acc_fake = tf.reduce_mean(tf.cast(y_bin_pred_fake == y_bin_true_fake, dtype=tf.float32))

        y_bin_pred_real = tf.cast(
            tf.nn.sigmoid(y_pred.real_descriminant.logits) > 0.5,
            dtype=tf.float32,
        )
        y_bin_true_real = tf.cast(descrim_t > 0.5, dtype=tf.float32)
        acc_real = tf.reduce_mean(tf.cast(y_bin_pred_real == y_bin_true_real, dtype=tf.float32))

        acc = (acc_fake + acc_real) / 2

        self.add_metric(
            descrim_fake_loss,
            aggregation="mean",
            name=coalsece_nm(self.prefix, "descriminator_fake_loss"),
        )
        self.add_metric(
            descrim_real_loss,
            aggregation="mean",
            name=coalsece_nm(self.prefix, "descriminator_real_loss"),
        )

        self.add_metric(
            acc_fake,
            aggregation="mean",
            name=coalsece_nm(self.prefix, "descriminator_fake_accuracy"),
        )
        self.add_metric(
            acc_real,
            aggregation="mean",
            name=coalsece_nm(self.prefix, "descriminator_real_accuracy"),
        )
        self.add_metric(
            acc, aggregation="mean", name=coalsece_nm(self.prefix, "descriminator_accuracy")
        )

        return descrim_real_loss + descrim_fake_loss


class GanLossNet(tf.keras.layers.Layer):
    def __init__(self, prefix=None, **kwargs):
        super().__init__(**kwargs)
        self.gen_lossnet = GanGeneratorLossNet(prefix=prefix)
        self.descrim_lossnet = GanDescriminationLossNet(prefix=prefix)
        self.prefix = prefix
        # assign the input to be the input of the parent model

    def call_fool_descriminator(
        self,
        y_true: Union[tf.Tensor, NamedTuple, tf.experimental.ExtensionType],
        y_pred: GanGenerativeNet.Output,
        training: bool = False,
    ):
        return self.gen_lossnet(y_true, y_pred, training)

    def call_tune_descriminator(
        self,
        y_true: Union[tf.Tensor, NamedTuple, tf.experimental.ExtensionType],
        y_pred: GanGenerativeNet.Output,
        training: bool = False,
    ):
        return self.descrim_lossnet(y_true, y_pred, training)

    def call(
        self,
        y_true: Union[tf.Tensor, NamedTuple, tf.experimental.ExtensionType],
        y_pred: GanNet.Output,
        training: bool = False,
    ):

        return (
            self.call_fool_descriminator(y_true, y_pred.generative, training),
            self.call_tune_descriminator(y_true, y_pred.descriminative, training),
        )

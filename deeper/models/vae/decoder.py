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


class VaeReconstructionNet(Layer):
    @dataclass
    class Config(Encoder.Config, DataSplitter.Config):
        def __post_init__(self):
            Encoder.Config.__post_init__()
            DataSplitter.Config.__post_init__()

    class Output(NamedTuple):
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

    @classmethod
    def from_config(cls, config: VaeReconstructionNet.Config, **kwargs):
        return cls(**config, **kwargs)

    @inits_args
    def __init__(
        self,
        output_regression_dimension: int,
        output_boolean_dimension: int,
        output_ordinal_dimension: Union[int, Sequence[int]],
        output_categorical_dimension: Union[int, Sequence[int]],
        decoder_embedding_dimensions: Tuple[int],
        embedding_activations=tf.nn.relu,
        bn_before: bool = False,
        bn_after: bool = False,
        recon_embedding_kernel_initializer="glorot_uniform",
        recon_embedding_bias_initializer="zeros",
        recon_latent_kernel_initialiazer="glorot_uniform",
        recon_latent_bias_initializer="zeros",
        recon_dropouut: Optional[float] = None,
        **kwargs,
    ):
        Layer.__init__(self, **kwargs)

        self.splitter = DataSplitter(
            DataSplitter.Config(
                output_regression_dimension,
                output_boolean_dimension,
                output_ordinal_dimension,
                output_categorical_dimension,
            )
        )
        (
            self.output_ordinal_dimension,
            self.output_ordinal_dimension_tot,
            self.output_categorical_dimension,
            self.output_categorical_dimension_tot,
            self.output_dim,
        ) = self.splitter.unpack_dimensions()

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

    def logits_to_actuals(
        self, output_logits_concat: tf.Tensor, training=False
    ) -> self.ReconstructionOutput:
        """Binary and categorical logits need to be converted into probs"""

        x_recon_logit = self.splitter(output_logits_concat, training)

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

        return self.Output(
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
    def call(
        self, x: VaeEncoderNet.Output, training=False
    ) -> VaeReconstructionNet.Output:
        return self.logits_to_actuals(
            self.graph_px_g_z.call(x.sample, training), training
        )

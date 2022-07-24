from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Activation

from typing import Tuple, Union, Optional, Sequence, NamedTuple
from dataclasses import dataclass
from pydantic import BaseModel

from deeper.utils.function_helpers.decorators import inits_args
from deeper.layers.encoder import Encoder
from deeper.layers.data_splitter import split_inputs, unpack_dimensions
from deeper.layers.data_splitter import DataSplitter, reduce_groups

from deeper.models.generalised_autoencoder.base import MultipleObjectiveDimensions
from deeper.models.vae.encoder import VaeEncoderNet
from deeper.utils.tf.experimental.extension_type import ExtensionTypeIterableMixin


from deeper.optimizers.automl.tunable_types import (
    TunableModelMixin,
    TunableBoolean,
    TunableActivation,
    OptionalTunableL1L2Regulariser,
    OptionalTunableDropout,
)


class VaeReconstructionNet(Layer):
    class Config(TunableModelMixin):
        output_dimensions: MultipleObjectiveDimensions
        decoder_embedding_dimensions: Sequence[int]
        embedding_activations: tf.keras.layers.Layer = TunableActivation("relu")
        bn_before: bool = TunableBoolean(False)
        bn_after: bool = False
        embedding_kernel_initializer: Union[
            str, tf.keras.initializers.Initializer
        ] = tf.initializers.glorot_uniform()
        embedding_bias_initializer: Union[
            str, tf.keras.initializers.Initializer
        ] = tf.initializers.zeros()
        embedding_kernel_regularizer: Optional[
            tf.keras.regularizers.Regularizer
        ] = OptionalTunableL1L2Regulariser(0.0, 0.0)
        embedding_bias_regularizer: Optional[
            tf.keras.regularizers.Regularizer
        ] = OptionalTunableL1L2Regulariser(0.0, 0.0)
        embedding_activity_regularizer: Optional[tf.keras.regularizers.Regularizer] = None
        recon_embedding_kernel_initializer: Union[
            str, tf.keras.initializers.Initializer
        ] = "glorot_uniform"
        recon_embedding_bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros"
        recon_latent_kernel_initialiazer: Union[
            str, tf.keras.initializers.Initializer
        ] = "glorot_uniform"
        recon_latent_bias_initializer: Union[str, tf.keras.initializers.Initializer] = "zeros"
        recon_input_dropout: Optional[float] = None
        recon_dropout: Optional[float] = None
        recon_kernel_regularizer: Optional[
            tf.keras.regularizers.Regularizer
        ] = OptionalTunableL1L2Regulariser(0.0, 0.0)
        recon_bias_regularizer: Optional[
            tf.keras.regularizers.Regularizer
        ] = OptionalTunableL1L2Regulariser(0.0, 0.0)
        recon_activity_regularizer: Optional[tf.keras.regularizers.Regularizer] = None
        binary_label_smoothing: float = 0.0

        class Config:
            arbitrary_types_allowed = True
            smart_union = True

    class Output(tf.experimental.ExtensionType, ExtensionTypeIterableMixin):
        hidden_logits: tf.Tensor
        regression: tf.Tensor
        logits_binary: tf.Tensor
        binary: tf.Tensor
        logits_ordinal_groups_concat: tf.Tensor
        logits_ordinal_groups: Tuple[tf.Tensor, ...]
        ord_groups_concat: tf.Tensor
        ord_groups: Tuple[tf.Tensor, ...]
        logits_categorical_groups_concat: tf.Tensor
        logits_categorical_groups: Tuple[tf.Tensor, ...]
        categorical_groups_concat: tf.Tensor
        categorical_groups: Tuple[tf.Tensor, ...]

    def __init__(
        self,
        config: VaeReconstructionNet.Config,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.splitter = DataSplitter(
            DataSplitter.Config(*config.output_dimensions.as_list()), **kwargs
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
            config.decoder_embedding_dimensions,
            activation=config.embedding_activations,
            bn_before=config.bn_before,
            bn_after=config.bn_after,
            embedding_kernel_initializer=config.recon_embedding_kernel_initializer,
            embedding_bias_initializer=config.recon_embedding_bias_initializer,
            latent_kernel_initialiazer=config.recon_latent_kernel_initialiazer,
            latent_bias_initializer=config.recon_latent_bias_initializer,
            input_dropout=config.recon_input_dropout,
            embedding_dropout=config.recon_dropout,
            embedding_kernel_regularizer=config.embedding_kernel_regularizer,
            embedding_bias_regularizer=config.embedding_bias_regularizer,
            embedding_activity_regularizer=config.embedding_activity_regularizer,
            latent_kernel_regularizer=config.recon_kernel_regularizer,
            latent_bias_regularizer=config.recon_bias_regularizer,
            latent_activity_regularizer=config.recon_activity_regularizer,
        )

    @tf.function
    def logits_to_actuals(
        self, output_logits_concat: tf.Tensor, training=False
    ) -> self.ReconstructionOutput:
        """Binary and categorical logits need to be converted into probs"""

        x_recon_logit = self.splitter(output_logits_concat, training)

        x_recon_bin = tf.nn.sigmoid(x_recon_logit.binary)
        x_recon_ord_groups = [tf.nn.sigmoid(x) for x in x_recon_logit.ordinal_groups]
        x_recon_ord_groups_concat = reduce_groups(tf.nn.softmax, x_recon_ord_groups)
        x_recon_cat_groups = [tf.nn.softmax(x) for x in x_recon_logit.categorical_groups]
        x_recon_cat_groups_concat = reduce_groups(tf.nn.softmax, x_recon_cat_groups)

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
    def call(self, x: VaeEncoderNet.Output, training=False) -> VaeReconstructionNet.Output:
        return self.logits_to_actuals(self.graph_px_g_z.call(x.sample, training), training)

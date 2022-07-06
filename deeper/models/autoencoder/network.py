from __future__ import annotations

import tensorflow as tf
from deeper.layers.data_splitter import unpack_dimensions
from deeper.layers.encoder import Encoder
from deeper.models.autoencoder.utils import AutoencoderTypeGetter
from deeper.models.vae.decoder import VaeReconstructionNet
from deeper.models.generalised_autoencoder.base import AutoencoderBase
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Activation, Layer
from deeper.utils.tf.experimental.extension_type import ExtensionTypeIterableMixin
from deeper.models.vae.base import MultipleObjectiveDimensions


class AutoencoderNet(AutoencoderBase):
    class Config(AutoencoderTypeGetter, AutoencoderBase.Config):
        class Config:
            arbitrary_types_allowed = True

    class EncoderOutputWrapper(tf.experimental.ExtensionType):
        sample: tf.Tensor

    class Output(tf.experimental.ExtensionType, ExtensionTypeIterableMixin, AutoencoderTypeGetter):
        latent: tf.Tensor
        reconstruction: VaeReconstructionNet.Output

    def __init__(
        self,
        config: AutoencoderNet.Config,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config

        # Input Dimension Calculation
        (
            self.input_ordinal_dimension,
            self.input_ordinal_dimension_tot,
            self.input_categorical_dimension,
            self.input_categorical_dimension_tot,
            self.input_dim,
        ) = unpack_dimensions(*config.input_dimensions.as_list())

        self.encoder = Encoder.from_config(
            Encoder.Config(
                latent_dim=config.latent_dim,
                activation=config.embedding_activations,
                embedding_dimensions=config.encoder_embedding_dimensions,
                **config.encoder_kwargs,
            )
        )

        self.decoder = VaeReconstructionNet(
            VaeReconstructionNet.Config(
                output_dimensions=config.output_dimensions,
                decoder_embedding_dimensions=config.decoder_embedding_dimensions,
                embedding_activations=config.embedding_activations,
                embedding_dimensions=config.decoder_embedding_dimensions,
                **config.decoder_kwargs,
            ),
            **kwargs,
        )

    @property
    def output_ordinal_dimension(self):
        return self.decoder.output_ordinal_dimension

    @property
    def output_ordinal_dimension_tot(self):
        return self.decoder.output_ordinal_dimension_tot

    @property
    def output_categorical_dimension(self):
        return self.decoder.output_categorical_dimension

    @property
    def output_categorical_dimension_tot(self):
        return self.decoder.output_categorical_dimension_tot

    @property
    def output_dim(self):
        return self.decoder.output_dim

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
    def split_outputs(self, y) -> SplitCovariates:
        return self.decoder.splitter(y)

    @tf.function
    def call(self, x, training=False) -> VaeNet.VaeNetOutput:

        x = tf.cast(x, dtype=self.dtype)
        # Encoder
        latent = self.encoder.call(x, training)
        # Decoder
        decoded = self.decoder.call(AutoencoderNet.EncoderOutputWrapper(sample=latent), training)

        return self.Output(latent, decoded)

    @classmethod
    def call_to_dict(self, result):
        return result._asdict()

    @tf.function
    def call_dict(self, x, training=False):
        return self.call_to_dict(self.call(x, training))

    @property
    def dim_latent(self):
        return self.encoder.latent_dimension

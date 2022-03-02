import tensorflow as tf
from tyiping import List, Union, Optional

from deeper.utils.function_helpers.decorators import inits_args

tfk = tf.keras
Layer = tfk.layers.Layer


class LstmEncoder(Layer, Scope):
    @inits_args
    def __init__(
        self,
        latent_dim: int,
        embedding_dimensions: List[int],
        activation: Union[str, tfk.layers.Activation],
        bn_before: bool = False,
        bn_after: bool = False,
        embedding_kernel_initializer: Union[str, tfk.initializers.Initializer] = "glorot_uniform",
        embedding_bias_initializer: Union[str, tfk.initializers.Initializer] = "zeros",
        latent_kernel_initialiazer: Union[str, tfk.initializers.Initializer] = "glorot_uniform",
        latent_bias_initializer: Union[str, tfk.initializers.Initializer] = "zeros",
        latent_return_sequences: bool = False,
        embedding_dropout: Optional[float] = None,
        opt_embedding_args: dict = {},
        opt_latent_args: dict = {},
        **kwargs
    ):
        Layer.__init__(self, **kwargs)

        # embeddings
        self.embeddings = []
        self.embeddings_bn_before = []
        self.embeddings_bn_after = []
        self.activation = activation
        self.bn_before = bn_before
        self.bn_after = bn_after
        self.dropout_rate = embedding_dropout
        self.dropout = []

        for i, em in enumerate(self.embedding_dimensions):
            self.embeddings.append(
                tfk.layers.LSTM(
                    units=em,
                    activation=None,
                    kernel_initializer=embedding_kernel_initializer,
                    bias_initializer=embedding_bias_initializer,
                    name="embedding_{}_dense".format(i),
                    return_sequences=True,
                    **opt_embedding_args
                )
            )
            if self.bn_before:
                self.embeddings_bn_before.append(
                    tfk.layers.BatchNormalization(
                        axis=-1,
                        name="embedding_{}_bn_before".format(i),
                        renorm=True,
                    )
                )
            else:
                self.embeddings_bn_before.append(None)

            if self.bn_after:
                self.embeddings_bn_after.append(
                    tfk.layers.BatchNormalization(
                        axis=-1,
                        name="embedding_{}_bn_after".format(i),
                        renorm=True,
                    )
                )
            else:
                self.embeddings_bn_after.append(None)

            if self.dropout_rate > 0.0 and self.dropout_rate is not None:
                self.dropout.append(tfk.layers.Dropout(self.dropout_rate))

        self.latent_bn = tfk.layers.BatchNormalization(axis=-1, name=self.v_name("latent_bn"))
        self.latent = tfk.layers.lstm(
            units=self.latent_dim,
            activation=None,
            kernel_initializer=latent_kernel_initialiazer,
            bias_initializer=latent_bias_initializer,
            name="latent_dense",
            return_sequences=latent_return_sequences,
            **opt_latent_args
        )

    @tf.function
    def call(self, inputs, training=False):
        """Define the computational flow"""
        x = inputs
        for em, bnb, bna, drp in zip(
            self.embeddings,
            self.embeddings_bn_before,
            self.embeddings_bn_after,
            self.dropout,
        ):
            x = em(x)
            if self.bn_before:
                x = bnb(x, training=training)
            x = self.activation(x)
            if self.bn_after:
                x = bna(x, training=training)
            if self.dropout_rate > 0.0:
                x = drp(x, training=training)
        x = self.latent(x)
        return x

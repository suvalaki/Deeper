import tensorflow as tf
from tensorflow.keras import initializers

from deeper.layers.encoder import Encoder

# from deeper.layers.conv2d_encoder import Conv2dEncoder, Conv2dDecoder
from deeper.utils.scope import Scope
from deeper.utils.function_helpers.decorators import inits_args

Layer = tf.keras.layers.Layer


class CategoricalEncoder(Layer, Scope):
    def __init__(
        self,
        latent_dimension,
        embedding_dimensions,
        embedding_activation=tf.nn.relu,
        var_scope="cat_encoder",
        bn_before=False,
        bn_after=False,
        epsilon=0.0,
        embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        embedding_bias_initializer=tf.initializers.zeros(),
        latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        latent_bias_initializer=tf.initializers.zeros(),
        embedding_dropout=0.0,
    ):
        Layer.__init__(self)
        Scope.__init__(self, var_scope)

        self.latent_dimension = latent_dimension
        self.embedding_dimensions = embedding_dimensions
        self.embedding_activation = embedding_activation
        self.bn_before = bn_before
        self.bn_after = bn_after
        self.epsilon = epsilon

        self.logits_encoder = Encoder(
            latent_dim=self.latent_dimension,
            embedding_dimensions=self.embedding_dimensions,
            activation=self.embedding_activation,
            var_scope=self.v_name("logits_encoder"),
            bn_before=self.bn_before,
            bn_after=self.bn_after,
            embedding_kernel_initializer=embedding_kernel_initializer,
            embedding_bias_initializer=embedding_bias_initializer,
            latent_kernel_initialiazer=latent_kernel_initialiazer,
            latent_bias_initializer=latent_bias_initializer,
            embedding_dropout=embedding_dropout,
        )

    @tf.function
    def call_logits(self, inputs, training=False):
        logits = self.logits_encoder(inputs, training)
        if self.epsilon > 0.0:
            maxval = np.log(1.0 - self.epsilon) - np.log(self.epsilon)
            logits = tf.compat.v2.clip_by_value(
                logits, -maxval, maxval, self.v_name("clipped")
            )
        return logits

    @tf.function
    def _prob(self, logits):
        prob = tf.nn.softmax(logits, axis=-1, name="prob")
        if self.epsilon > 0.0:
            prob = tf.compat.v2.clip_by_value(
                prob,
                self.epsilon,
                1 - self.epsilon,
                self.v_name("prob_clipepd"),
            )
        return prob

    @tf.function  # (experimental_relax_shapes=True)
    def call(self, inputs, y=None, training=False):
        logits = self.logits_encoder(inputs, training)
        probs = self._prob(logits)
        if y is not None:
            ent = tf.nn.softmax_cross_entropy_with_logits(
                labels=y, logits=logits, name="entropy"
            )
        else:
            ent = tf.nn.softmax_cross_entropy_with_logits(
                labels=probs, logits=logits, name="entropy"
            )
        return logits, probs

    @tf.function
    def prob(self, inputs, training=False):
        logits, probs = self.call(inputs, training)
        return probs

    @tf.function
    def entropy(self, x, y, training=False):
        logits = self.call_logits(x, training)
        ent = tf.nn.softmax_cross_entropy_with_logits(
            labels=y, logits=logits, name="entropy"
        )
        return ent


"""
class CategoricalEncoder2DBase(Layer, Scope):

    @init_args
    def __init__(
        self, 
        encoder, 
        depth,
        latent_dim, 
        kernel_size,
        filters:list,
        strides=(3,3),
        padding='SAME',
        embedding_activation=tf.nn.relu,
        var_scope='cat_encoder',
        bn_before=False,
        bn_after=False,
        epsilon=0.0,
        embedding_kernel_initializer=tf.initializers.glorot_uniform(),
        embedding_bias_initializer=tf.initializers.zeros(),
        latent_kernel_initialiazer=tf.initializers.glorot_uniform(),
        latent_bias_initializer=tf.initializers.zeros(),
        embedding_dropout=0.0,
    ):
        Layer.__init__(self)
        Scope.__init__(self, var_scope)

        self.logits_encoder = encoder(
            depth, latent_dim, kernel, filters, strides, padding, 
            embedding_activation, self.v_name('logits_encoder'), 
            bn_before, bn_after,
            embedding_kernel_initializer, embedding_bias_initializer,
            latent_kernel_initialiazer, latent_bias_initializer,
            embedding_dropout
        )


    @tf.function
    def call_logits(self, inputs, training=False):
        logits = self.logits_encoder(inputs, training)
        if self.epsilon > 0.0:
            maxval = np.log(1.0 - self.epsilon) - np.log(self.epsilon)
            logits = tf.compat.v2.clip_by_value(
                logits, -maxval, maxval, self.v_name("clipped"))
        return logits


    @tf.function
    def _prob(self, logits):
        prob = tf.nn.softmax(logits, axis=-1, name='prob')
        if self.epsilon > 0.0:
            prob = tf.compat.v2.clip_by_value(
                prob, self.epsilon, 1-self.epsilon, self.v_name('prob_clipepd')
            )
        return prob


    @tf.function 
    def call(self, inputs, y=None, training=False, return_dict=False):
        logits = self.logits_encoder(inputs, training)
        probs = self._prob(logits)
        if y is not None:
            ent = tf.nn.softmax_cross_entropy_with_logits(
                labels=y, 
                logits=logits, 
                name='entropy'
            )
        else:
            ent = tf.nn.softmax_cross_entropy_with_logits(
                labels=probs, 
                logits=logits, 
                name='entropy'
            )
        if not return_dict:
            return logits, probs
        else: 
            return {'logits': logits, 'probs':probs, 'entropy': ent}


    @tf.function
    def prob(self, inputs, training=False):
        logits, probs = self.call(inputs, training)
        return probs


    @tf.function
    def entropy(self, x, y, training=False):
        logits = self.call_logits(x, training)
        ent = tf.nn.softmax_cross_entropy_with_logits(
            labels=y, 
            logits=logits,
            name='entropy'
        )
        return ent
"""

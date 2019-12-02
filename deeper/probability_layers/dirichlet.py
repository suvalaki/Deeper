import tensorflow as tf
from deeper.utils.scope import Scope
from deeper.layers.encoder import Encoder

tfk = tf.keras
Layer = tfk.layers.Layer


class DirichletGenerator:
    @tf.function
    def _rejection_step(self, c, d):

        u = tf.random.uniform()
        z = tf.random.normal()

        v = tf.math.pow(1 + c * z, 3)
        sample = d * v
        b1 = -1 / c
        b2 = 0.5 * tf.math.square(z) + d - (d * v) + (d * tf.math.log(v))

        return u, z, v, sample, b1, b2

    @tf.function
    def _rejection_sample(self, alpha):

        d = alpha - 1 / 3
        c = 1 / tf.math.sqrt(9 * d)

        u, z, v, sample, b1, b2 = self._rejection_step(c, d)
        while z > b1 or tf.math.log(u) > b2:
            u, z, v, sample, b1, b2 = self._rejection_step(c, d)

        return sample


class DirichletEncoder(Layer, Scope, DirichletGenerator):
    def __init__(
        self,
        latent_dimension,
        embedding_dimensions,
        embedding_activations=tf.nn.relu,
        var_scope="dirichlet_encoder",
        bn_before=False,
        bn_after=False,
        epsilon=0.0,
        embedding_mu_kernel_initializer=tf.initializers.glorot_normal(),
        embedding_mu_bias_initializer=tf.initializers.zeros(),
        latent_mu_kernel_initialiazer=tf.initializers.glorot_normal(),
        latent_mu_bias_initializer=tf.initializers.zeros(),
        embedding_var_kernel_initializer=tf.initializers.glorot_normal(),
        embedding_var_bias_initializer=tf.initializers.zeros(),
        latent_var_kernel_initialiazer=tf.initializers.glorot_normal(),
        latent_var_bias_initializer=tf.initializers.zeros(),
    ):

        Layer.__init__(self)
        Scope.__init__(self, var_scope)

        self.latent_dimension = latent_dimension
        self.embedding_dimensions = embedding_dimensions
        self.embedding_activation = embedding_activations
        self.bn_before = bn_before
        self.bn_after = bn_after
        self.epsilon = epsilon

        self.encoder = Encoder(
            latent_dim=self.latent_dimension,
            embedding_dimensions=self.embedding_dimensions,
            activation=self.embedding_activation,
            var_scope=self.v_name("encoder"),
            bn_before=self.bn_before,
            bn_after=self.bn_after,
            embedding_kernel_initializer=embedding_mu_kernel_initializer,
            embedding_bias_initializer=embedding_mu_bias_initializer,
            latent_kernel_initialiazer=latent_mu_kernel_initialiazer,
            latent_bias_initializer=latent_mu_bias_initializer,
        )

    @tf.function
    def call_parameters(self, x, training=False):
        alpha = self.encoder(x, training)
        return alpha

    @tf.function
    def sample(self, x, training=False):
        alpha = self.encoder(x, training)
        sample = self._rejection_sample(alpa)
        return sample

    @tf.function
    def call(self, x, training=False):
        alpha = self.encoder(x, training)
        sample = self._rejection_sample(alpha)
        return alpha, sample

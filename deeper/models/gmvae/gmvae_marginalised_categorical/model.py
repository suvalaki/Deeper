import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.eager import context
import numpy as np
import datetime

tfd = tfp.distributions
tfk = tf.keras

Model = tfk.Model

# tf.enable_eager_execution()

import numpy as np

steps = 0

class Encoder(Model):
    def __init__(self, latent_dim, embedding_dimensions):
        Model.__init__(self)
        self.latent_dim = latent_dim
        self.em_dim = embedding_dimensions

        # embeddings
        self.embeddings = []
        self.embeddings_bn = []
        
        for i,em in enumerate(self.em_dim):
            with tf.name_scope('embedding_{}'.format(i)):
                self.embeddings.append(
                    tfk.layers.Dense(
                        units=em,
                        activation=None,
                        use_bias=True,
                        kernel_initializer=tf.initializers.glorot_normal(seed=None),
                        bias_initializer=tf.initializers.glorot_normal(seed=None),
                        name='layer'
                    )
                )
                self.embeddings_bn.append(tfk.layers.BatchNormalization(axis=-1))

        with tf.name_scope('latent'):
            self.latent_bn = tfk.layers.BatchNormalization(axis=-1)
            self.latent = tfk.layers.Dense(
                units=self.latent_dim,
                activation=None,
                use_bias=True,
                kernel_initializer=tf.initializers.glorot_normal(seed=None),
                bias_initializer=tf.initializers.glorot_normal(seed=None),
            )

    @tf.function  # (autograph=False)
    def call(self, inputs, training=False):
        """Define the computational flow"""
        x = tf.cast(inputs, tf.float32)
        for em, bn in zip(self.embeddings, self.embeddings_bn):
            x = em(x)
            x = tf.nn.relu(x)
            x = bn(x, training=training)
            
        x = self.latent(x)
        #x = self.latent_bn(x, training=training)
        #x = tf.nn.tanh(x)
        return x


class SoftmaxEncoder(Model):
    def __init__(self, latent_dim, embedding_dimensions):
        Model.__init__(self)
        self.latent_dim = latent_dim
        self.embedding_dimensions = embedding_dimensions

        self.logits = Encoder(self.latent_dim, self.embedding_dimensions)

    @tf.function  # (autograph=False)
    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)
        eps = 0.01
        maxval = np.log(1.0 - eps) - np.log(eps)
        logits = tf.compat.v2.clip_by_value(
            self.logits(x, training), -maxval, maxval, "clipped"
        )
        #prob =  tf.nn.softmax(logits)
        prob = tf.nn.softmax(logits)
        return logits, prob





class NormalEncoder(Model):
    def __init__(self, latent_dim, embedding_dimensions):
        Model.__init__(self)
        self.latent_dim = latent_dim
        self.embedding_dimensions = embedding_dimensions
        self.bn_mu = tfk.layers.BatchNormalization(axis=-1)
        self.mu = Encoder(self.latent_dim, self.embedding_dimensions)
        self.logvar = Encoder(self.latent_dim, self.embedding_dimensions)
        self.bn_logvar = tfk.layers.BatchNormalization(axis=-1)
        self.sample = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Normal(t[0], tf.exp(t[1])),
            convert_to_tensor_fn=lambda s: s.sample(1),
        )

    @tf.function
    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)
        mu = self.mu(x, training)
        #mu = self.bn_mu(mu, training)
        logvar = self.logvar(x, training) + 1e-5
        #logvar = self.bn_logvar(logvar, training)
        #dist = self.sample((mu, logvar))
        
        # reparmeterisation trick
        r_norm = tf.random.normal( tf.shape(mu), mean=0., stddev=1.)
        sample = mu + r_norm * tf.math.sqrt(tf.exp(logvar))

        # Metrics for loss
        logprob = self.log_normal(sample, mu, tf.exp(logvar))
        prob = tf.exp(logprob)

        return sample, logprob, prob

      
    @staticmethod
    def log_normal(x, mu, var, eps=0.0, axis=-1):
        if eps > 0.0:
            var = tf.add(var, eps, name='clipped_var')
        return -0.5 * tf.reduce_sum(
            tf.math.log(2 * np.pi) + tf.math.log(var) + tf.square(x - mu) / var, axis)




class NormalDecoder(Model):
    def __init__(self, latent_dim, embedding_dimensions):
        Model.__init__(self)
        self.latent_dim = latent_dim
        self.embedding_dimensions = embedding_dimensions

        self.mu = Encoder(self.latent_dim, self.embedding_dimensions)
        self.logvar = Encoder(self.latent_dim, self.embedding_dimensions)
        self.sample = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Normal(t[0], tf.exp(t[1])),
            convert_to_tensor_fn=lambda s: s.sample(),
        )

    @tf.function  # (autograph=False)
    def call(self, inputs, outputs, training=False, var=None):
        x = tf.cast(inputs, tf.float32)
        mu = self.mu.call(x, training)
        # logvar = self.logvar(x, training)
        if var is not None:
            logvar = (
                tf.exp(tf.cast(self.logvar(x, training), tf.float32)) + 1e-5
            )
        else:
            logvar = (
                tf.fill(tf.shape(outputs), tf.cast(1.0, tf.float32)) + 1e-5
            )

        # Metrics for loss
        logprob = self.log_normal(mu, outputs, tf.exp(logvar))
        prob = tf.exp(logprob)

        return 0, logprob, prob

      
    @staticmethod
    def log_normal(x, mu, var, eps=0.0, axis=-1):
        if eps > 0.0:
            var = tf.add(var, eps, name='clipped_var')
        return -0.5 * tf.reduce_sum(
            tf.math.log(2 * np.pi) + tf.math.log(var) + tf.square(x - mu) / var, axis)



class SigmoidDecoder(Model):
    def __init__(self, latent_dim, embedding_dimensions):
        Model.__init__(self)
        self.latent_dim = latent_dim
        self.embedding_dimensions = embedding_dimensions
        self.mu = Encoder(self.latent_dim, self.embedding_dimensions)
        self.sample = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Bernoulli(logits=t),
            convert_to_tensor_fn=lambda s: s.sample(),
        )

    @tf.function  # (autograph=False)
    def call(self, inputs, outputs, training=False):
        x = tf.cast(inputs, tf.float32)
        logit = self.mu.call(x, training)

        eps = 0.01
        if eps > 0.0:
            max_val = np.log(1.0 - eps) - np.log(eps)
            logit = tf.compat.v2.clip_by_value(
                logit, -max_val, max_val, "clipped"
            )

        logprob = self.log_bernoulli_with_logits(tf.cast(outputs, tf.float32), tf.cast(logit, tf.float32))
        prob = tf.exp(logprob)

        return logit[None, :, :], logprob, prob

    @staticmethod
    def log_bernoulli_with_logits(x, logits, eps=0.0, axis=-1):
        if eps > 0.0:
            max_val = np.log(1.0 - eps) - np.log(eps)
            logits = tf.clip_by_value(logits, -max_val, max_val,
                                    name='clipped_logit')
        return -tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x), axis)

class MarginalAutoEncoder(Model):
    def __init__(
        self, input_dimension, embedding_dimensions, latent_dim, kind="binary"
    ):
        Model.__init__(self)
        self.in_dim = input_dimension
        self.la_dim = latent_dim
        self.em_dim = embedding_dimensions
        self.kind = kind

        with tf.name_scope('graph_qz_g_xy'):
            self.graphs_qz_g_xy = NormalEncoder(self.la_dim, self.em_dim)
        with tf.name_scope('graph_pz_g_y'):
            self.graphs_pz_g_y = NormalDecoder(
                self.la_dim, [self.la_dim]
            )
        with tf.name_scope('graph_px_g_y'):
            if self.kind == "binary":
                self.graphs_px_g_zy = SigmoidDecoder(
                    self.in_dim, self.em_dim[::-1]
                )
            #else:
            #    self.graphs_px_g_zy = NormalDecoder(self.in_dim, self.em_dim[::-1])

    # @tf.function#
    def call(self, x, y, training=False):

        xy = tf.concat([tf.cast(x, tf.float32), tf.cast(y, tf.float32)], axis=-1)
        (
            qz_g_xy__sample,
            qz_g_xy__logprob,
            qz_g_xy__prob,
        ) = self.graphs_qz_g_xy.call(xy, training)
        (
            pz_g_y__sample,
            pz_g_y__logprob,
            pz_g_y__prob,
        ) = self.graphs_pz_g_y.call(y, qz_g_xy__sample, training, var=True)
        (
            px_g_zy__sample,
            px_g_zy__logprob,
            px_g_zy__prob,
        ) = self.graphs_px_g_zy.call(qz_g_xy__sample, x, training)

        return (
            qz_g_xy__sample,
            qz_g_xy__logprob,
            qz_g_xy__prob,
            pz_g_y__sample,
            pz_g_y__logprob,
            pz_g_y__prob,
            px_g_zy__sample,
            px_g_zy__logprob,
            px_g_zy__prob,
        )

    # @tf.function
    def sample(self, samples, x, y, training=False):
        with tf.device("/gpu:0"):
            result = [self.call(x, y, training) for j in range(samples)]
            result_pivot = list(zip(*result))
        return result_pivot

    @staticmethod
    def mc_stack_mean(x):
        return tf.identity(tf.stack(x, 0) / len(x))

    # @tf.function
    def monte_carlo_estimate(self, samples, x, y, training=False):
        return [
            self.mc_stack_mean(z)
            for z in self.sample(samples, x, y, training=False)
        ]


class Gmvae(Model):
    def __init__(
        self,
        components,
        input_dimension,
        embedding_dimensions,
        latent_dimensions,
        kind="binary",
        monte_carlo_samples=1,
        beta=0.01,
        lmbda=0.5,
        learning_rate=0.01,
        gradient_clip=1.0,
    ):

        # instatiate
        Model.__init__(self)

        self.kind = kind
        self.k = components
        self.in_dim = input_dimension
        self.em_dim = embedding_dimensions
        self.la_dim = latent_dimensions
        self.mc_sam = monte_carlo_samples

        self.epochs = 0
        self.beta = beta
        self.lmbda = lmbda
        self.gradient_clip = gradient_clip

        self.learning_rate = learning_rate

        # instantiate all variables in the graph
        self.graph_qy_g_x = SoftmaxEncoder(self.k, self.em_dim)
        self.marginal_autoencoder = [
            MarginalAutoEncoder(
                self.in_dim, self.em_dim, self.la_dim, self.kind
            )
            for i in range(self.k)
        ]

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def call(self, inputs, training=False):

        x = inputs
        y_ = tf.fill(tf.stack([tf.shape(x)[0], self.k]), 0.0)
        py = tf.fill(tf.shape(y_), 1 / self.k, name="prob")

        with tf.name_scope('graph_qy'):
            qy_g_x__logit, qy_g_x__prob = self.graph_qy_g_x(x, training)

        mc_qz_g_xy__sample = [None] * self.k
        mc_qz_g_xy__logprob = [None] * self.k
        mc_qz_g_xy__prob = [None] * self.k

        mc_pz_g_y__sample = [None] * self.k
        mc_pz_g_y__logprob = [None] * self.k
        mc_pz_g_y__prob = [None] * self.k

        mc_px_g_zy__sample = [None] * self.k
        mc_px_g_zy__logprob = [None] * self.k
        mc_px_g_zy__prob = [None] * self.k

        for i in range(self.k):
            with tf.name_scope('mixture_{}'.format(i)):
                y = tf.add(
                    y_,
                    tf.constant(
                        np.eye(self.k)[i],
                        dtype=tf.float32,
                        name="y_one_hot_{}".format(i),
                    ),
                    name="hot_at_{}".format(i),
                )

                y = tf.cast(y, tf.float32)

                (
                    mc_qz_g_xy__sample[i],
                    mc_qz_g_xy__logprob[i],
                    mc_qz_g_xy__prob[i],
                    mc_pz_g_y__sample[i],
                    mc_pz_g_y__logprob[i],
                    mc_pz_g_y__prob[i],
                    mc_px_g_zy__sample[i],
                    mc_px_g_zy__logprob[i],
                    mc_px_g_zy__prob[i],
                ) = self.marginal_autoencoder[i].monte_carlo_estimate(
                    self.mc_sam, x, y, training
                )

        return (
            py,
            qy_g_x__prob,
            mc_qz_g_xy__sample,
            mc_qz_g_xy__logprob,
            mc_qz_g_xy__prob,
            mc_pz_g_y__sample,
            mc_pz_g_y__logprob,
            mc_pz_g_y__prob,
            mc_px_g_zy__sample,
            mc_px_g_zy__logprob,
            mc_px_g_zy__prob,
        )

    # @tf.function
    def entropy_fn(self, inputs, training=False):
        (
            py,
            qy_g_x,
            mc_qz_g_xy__sample,
            mc_qz_g_xy__logprob,
            mc_qz_g_xy__prob,
            mc_pz_g_y__sample,
            mc_pz_g_y__logprob,
            mc_pz_g_y__prob,
            mc_px_g_zy__sample,
            mc_px_g_zy__logprob,
            mc_px_g_zy__prob,
        ) = self.call(inputs, training=training)

        # reconstruction
        recon = tf.add_n(
            [
                tf.cast(qy_g_x[:, i], tf.float32)
                * tf.cast(mc_px_g_zy__logprob[i], tf.float32)
                for i in range(self.k)
            ]
        )

        # z_entropy
        z_entropy = tf.add_n(
            [
                tf.cast(qy_g_x[:, i], tf.float32)
                * (
                    tf.cast(mc_pz_g_y__logprob[i], tf.float32)
                    - tf.cast(mc_qz_g_xy__logprob[i], tf.float32)
                )
                for i in range(self.k)
            ]
        )

        # y_entropy
        y_entropy = tf.reduce_sum(
            tf.cast(qy_g_x, tf.float32)
            * (
                tf.math.log(tf.cast(py, tf.float32))
                - tf.math.log(tf.cast(qy_g_x, tf.float32))
            ),
            axis=-1,
        )

        # elbo = recon + z_entropy + y_entropy
        return recon, z_entropy, y_entropy

    # @tf.function#(autograph=False)
    def elbo(self, inputs, training=False):
        recon, z_entropy, y_entropy = self.entropy_fn(inputs, training)
        return recon + z_entropy + y_entropy

    # @tf.function#(autograph=False)
    def loss_fn(self, inputs, training=False):
        return -self.elbo(inputs, training)

    def even_mixture_loss(self, inputs, training=False):
        (
            py,
            qy_g_x,
            mc_qz_g_xy__sample,
            mc_qz_g_xy__logprob,
            mc_qz_g_xy__prob,
            mc_pz_g_y__sample,
            mc_pz_g_y__logprob,
            mc_pz_g_y__prob,
            mc_px_g_zy__sample,
            mc_px_g_zy__logprob,
            mc_px_g_zy__prob,
        ) = self.call(inputs, training=training)

        # reconstruction
        recon = tf.add_n(
            [
                tf.cast(py[:, i], tf.float32)
                * tf.cast(mc_px_g_zy__logprob[i], tf.float32)
                for i in range(self.k)
            ]
        )

        # z_entropy
        z_entropy = tf.add_n(
            [
                tf.cast(py[:, i], tf.float32)
                * (
                    tf.cast(mc_pz_g_y__logprob[i], tf.float32)
                    - tf.cast(mc_qz_g_xy__logprob[i], tf.float32)
                )
                for i in range(self.k)
            ]
        )

        return -(recon + z_entropy)

    # @tf.function#(autograph=False)
    def train_step(self, x, tenorboard=False):

        if tenorboard:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/gradient_tape/train'

            writer = tf.summary.create_file_writer(train_log_dir)


        # for x in dataset:
        # Tensorflow dataset is iterable in eager mode
        with tf.device("/gpu:0"):
            with tf.GradientTape() as tape:
                loss = tf.reduce_sum(self.loss_fn(x, training=True))
            # Update ops for batch normalization
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):

        with tf.device("/gpu:0"):
            gradients = tape.gradient(loss, self.trainable_variables)
            # Clipping
            gradients = [
                None
                if gradient is None
                else tf.clip_by_value(
                    gradient, -self.gradient_clip, self.gradient_clip
                )
                for gradient in gradients
            ]

            if tenorboard:
                with writer.as_default():
                    for gradient, variable in zip(gradients, self.trainable_variables):
                        global steps
                        steps = steps + 1
                        tf.summary.experimental.set_step(steps)
                        stp = tf.summary.experimental.get_step()
                        tf.summary.histogram("gradients/" + variable.name, tf.nn.l2_normalize(gradient), step=stp)
                        tf.summary.histogram("variables/" + variable.name, tf.nn.l2_normalize(variable), step=stp)
                    writer.flush()

        with tf.device("/gpu:0"):
            self.optimizer.apply_gradients(
                zip(gradients, self.trainable_variables)
            )

    def pretrain_step(self, x):
        # for x in dataset:
        # Tensorflow dataset is iterable in eager mode

        with tf.device("/gpu:0"):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(self.even_mixture_loss(x, training=True))
            # Update ops for batch normalization
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):

        with tf.device("/gpu:0"):
            gradients = tape.gradient(loss, self.trainable_variables)
            # Clipping
            gradients = [
                None
                if gradient is None
                else tf.clip_by_value(
                    gradient, -self.gradient_clip, self.gradient_clip
                )
                for gradient in gradients
            ]
        with tf.device("/gpu:0"):
            self.optimizer.apply_gradients(
                zip(gradients, self.trainable_variables)
            )

    # @tf.function#(autograph=False)
    def predict(self, x, training=False):
        qy_g_x__logit, qy_g_x__prob = self.graph_qy_g_x(x, training)
        return qy_g_x__prob

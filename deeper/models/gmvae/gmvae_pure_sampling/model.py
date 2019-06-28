import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfk = tf.keras

Model = tfk.Model

# tf.enable_eager_execution()

import numpy as np


class Encoder:
    def __init__(self, latent_dim, embedding_dimensions):
        # Model.__init__(self,)
        self.latent_dim = latent_dim
        self.em_dim = embedding_dimensions

        # embeddings
        self.embeddings = []
        self.embeddings_bn = []
        for em in self.em_dim:
            self.embeddings.append(
                tfk.layers.Dense(
                    units=em,
                    activation=tf.nn.tanh,
                    use_bias=False,
                    kernel_initializer=tfk.initializers.he_normal(seed=None),
                    bias_initializer=tfk.initializers.Zeros(),
                )
            )
            self.embeddings_bn.append(tfk.layers.BatchNormalization(axis=-1))

        self.latent = tfk.layers.Dense(
            units=self.latent_dim,
            activation=None,
            use_bias=False,
            kernel_initializer=tfk.initializers.he_normal(seed=None),
            bias_initializer=tfk.initializers.Zeros(),
        )

    @tf.function
    def call(self, inputs, training=False):
        """Define the computational flow"""
        x = tf.cast(inputs, tf.float64)
        for em, bn in zip(self.embeddings, self.embeddings_bn):
            x = em(x)
            x = bn(x, training=training)
        x = self.latent(x)
        return x


class GumbleSoftmaxEncoder:
    def __init__(self, latent_dim, embedding_dimensions):
        # Model.__init__(self)
        self.latent_dim = latent_dim
        self.embedding_dimensions = embedding_dimensions
        self.logits = Encoder(self.latent_dim, self.embedding_dimensions)
        self.sample = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfd.RelaxedOneHotCategorical(
                t[0], tf.exp(t[1])
            ),
            convert_to_tensor_fn=lambda s: s.sample(1),
        )

    @tf.function
    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float64)
        logits = self.logits(x)
        prob = tf.nn.softmax(logits)
        return logits, prob


class NormalEncoder:
    def __init__(self, latent_dim, embedding_dimensions):
        # Model.__init__(self)
        self.latent_dim = latent_dim
        self.embedding_dimensions = embedding_dimensions

        self.mu = Encoder(self.latent_dim, self.embedding_dimensions)
        self.logvar = Encoder(self.latent_dim, self.embedding_dimensions)
        self.sample = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfd.MultivariateNormalDiag(
                t[0], tf.exp(t[1])
            ),
            convert_to_tensor_fn=lambda s: s.sample(1),
        )

    @tf.function
    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float64)
        mu = self.mu(x, training)
        logvar = self.logvar(x, training)
        dist = self.sample((mu, logvar))
        sample = dist.sample(1)

        # Metrics for loss
        logprob = dist.log_prob(sample)
        prob = dist.prob(sample)

        return sample, logprob, prob


class NormalDecoder:
    def __init__(self, latent_dim, embedding_dimensions):
        # Model.__init__(self)
        self.latent_dim = latent_dim
        self.embedding_dimensions = embedding_dimensions

        self.mu = Encoder(self.latent_dim, self.embedding_dimensions)
        self.logvar = Encoder(self.latent_dim, self.embedding_dimensions)
        self.sample = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfd.MultivariateNormalDiag(
                t[0], tf.exp(t[1])
            ),
            convert_to_tensor_fn=lambda s: s.sample(),
        )

    @tf.function
    def call(self, inputs, outputs, training=False, var=None):
        x = tf.cast(inputs, tf.float64)
        mu = self.mu.call(x, training)
        # logvar = self.logvar(x, training)
        if var is not None:
            logvar = tf.exp(tf.cast(self.logvar(x, training), tf.float64))
        else:
            logvar = tf.fill(tf.shape(outputs), tf.cast(1.0, tf.float64))
        dist = self.sample((tf.cast(outputs, tf.float64), logvar))
        # sample = dist.sample(1)
        # Metrics for loss
        # import pdb; pdb.set_trace()
        logprob = dist.log_prob(tf.cast(mu[:, :], tf.float64))
        prob = dist.prob(tf.cast(mu[:, :], tf.float64))

        return mu[None, :, :], logprob, prob


class SigmoidDecoder:
    def __init__(self, latent_dim, embedding_dimensions):
        # Model.__init__(self)
        self.latent_dim = latent_dim
        self.embedding_dimensions = embedding_dimensions
        self.mu = Encoder(self.latent_dim, self.embedding_dimensions)
        self.sample = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Bernoulli(logits=t),
            convert_to_tensor_fn=lambda s: s.sample(),
        )

    @tf.function
    def call(self, inputs, outputs, training=False):
        x = tf.cast(inputs, tf.float64)
        logit = self.mu.call(x, training)
        # logvar = self.logvar(x, training)
        # dist = self.sample(tf.cast(outputs, tf.float64),)
        # sample = dist.sample(1)
        # Metrics for loss
        # import pdb; pdb.set_trace()
        # logprob = dist.log_prob(tf.cast(mu[:,:],tf.float64))
        # prob = dist.prob(tf.cast(mu[:,:],tf.float64))
        eps = 0.01
        if eps > 0.0:
            max_val = np.log(1.0 - eps) - np.log(eps)
            logit = tf.clip_by_value(logit, -max_val, max_val)

        logprob = -tf.cast(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logit, labels=outputs[None, :, :]
                ),
                # outputs[None,:,:] * tf.log(tf.nn.sigmoid(logit)),
                -1,
            ),
            tf.float64,
        )

        prob = tf.exp(logprob)

        return logit[None, :, :], logprob, prob


class Gmvae:
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
    ):

        # instatiate
        # Model.__init__(self,)

        self.kind = kind
        self.k = components
        self.in_dim = input_dimension
        self.em_dim = embedding_dimensions
        self.la_dim = latent_dimensions
        self.mc_sam = monte_carlo_samples

        self.epochs = 0
        self.beta = beta
        self.lmbda = lmbda

        self.learning_rate = learning_rate

        # instantiate all variables in the graph

        self.graph_qy_g_x = GumbleSoftmaxEncoder(self.k, self.em_dim)
        self.graphs_qz_g_xy = NormalEncoder(self.la_dim, self.em_dim)
        self.graphs_pz_g_y = NormalDecoder(self.la_dim, [int(self.la_dim // 2)])

        if self.kind == "binary":
            self.graphs_px_g_zy = SigmoidDecoder(self.in_dim, self.em_dim[::-1])
        else:
            self.graphs_px_g_zy = NormalDecoder(self.in_dim, self.em_dim[::-1])

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    @staticmethod
    @tf.function
    def mc_stack_mean(x):
        return tf.identity(tf.stack(x, 0) / len(x))

    @tf.function
    def call(self, inputs, training=False):
        x = inputs

        y_ = tf.fill(tf.stack([tf.shape(x)[0], self.k]), 0.0)
        py = tf.fill(tf.shape(y_), 1 / self.k, name="prob")

        qy_g_x__logit, qy_g_x__prob = self.graph_qy_g_x(x, training)

        qz_g_xy__sample = [None] * self.mc_sam
        qz_g_xy__logprob = [None] * self.mc_sam
        qz_g_xy__prob = [None] * self.mc_sam

        pz_g_y__sample = [None] * self.mc_sam
        pz_g_y__logprob = [None] * self.mc_sam
        pz_g_y__prob = [None] * self.mc_sam

        px_g_zy__sample = [None] * self.mc_sam
        px_g_zy__logprob = [None] * self.mc_sam
        px_g_zy__prob = [None] * self.mc_sam

        mc_qz_g_xy__sample = None
        mc_qz_g_xy__logprob = None
        mc_qz_g_xy__prob = None

        mc_pz_g_y__sample = None
        mc_pz_g_y__logprob = None
        mc_pz_g_y__prob = None

        mc_px_g_zy__sample = None
        mc_px_g_zy__logprob = None
        mc_px_g_zy__prob = None

        for j in range(self.mc_sam):
            y = tf.add(
                y_,
                tf.constant(
                    np.eye(self.k), dtype=tf.float32, name="y_one_hot".format(i)
                ),
            )
            y = tf.cast(y, tf.float64)

            xy = tf.concat([x, y], axis=-1)
            (
                qz_g_xy__sample[j],
                qz_g_xy__logprob[j],
                qz_g_xy__prob[j],
            ) = self.graphs_qz_g_xy.call(xy, training)
            (
                pz_g_y__sample[j],
                pz_g_y__logprob[j],
                pz_g_y__prob[j],
            ) = self.graphs_pz_g_y.call(y, qz_g_xy__sample[j], training, var=True)

            (
                px_g_zy__sample[j],
                px_g_zy__logprob[j],
                px_g_zy__prob[j],
            ) = self.graphs_px_g_zy.call(qz_g_xy__sample[j], inputs, training)

        # Monte Carlo
        mc_qz_g_xy__sample = self.mc_stack_mean(qz_g_xy__sample)
        # import pdb; pdb.set_trace()
        mc_qz_g_xy__logprob = self.mc_stack_mean(qz_g_xy__logprob)
        mc_qz_g_xy__prob = self.mc_stack_mean(qz_g_xy__prob)

        mc_pz_g_y__sample = self.mc_stack_mean(pz_g_y__sample)
        mc_pz_g_y__logprob = self.mc_stack_mean(pz_g_y__logprob)
        mc_pz_g_y__prob = self.mc_stack_mean(pz_g_y__prob)

        mc_px_g_zy__sample = self.mc_stack_mean(px_g_zy__sample)
        mc_px_g_zy__logprob = self.mc_stack_mean(px_g_zy__logprob)
        mc_px_g_zy__prob = self.mc_stack_mean(px_g_zy__prob)

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

    @tf.function
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
                tf.cast(qy_g_x[:, i], tf.float64)
                * tf.cast(mc_px_g_zy__logprob, tf.float64)
                for i in range(self.k)
            ]
        )

        # z_entropy
        z_entropy = tf.add_n(
            [
                tf.cast(qy_g_x[:, i], tf.float64)
                * (
                    tf.cast(mc_pz_g_y__logprob, tf.float64)
                    - tf.cast(mc_qz_g_xy__logprob, tf.float64)
                )
                for i in range(self.k)
            ]
        )

        # y_entropy
        y_entropy = tf.reduce_sum(
            tf.cast(qy_g_x, tf.float64)
            * (
                tf.math.log(tf.cast(py, tf.float64))
                - tf.math.log(tf.cast(qy_g_x, tf.float64))
            ),
            axis=-1,
        )

        # elbo = recon + z_entropy + y_entropy
        return recon, z_entropy, y_entropy

    @tf.function
    def elbo(self, inputs, training=False):
        recon, z_entropy, y_entropy = self.entropy_fn(inputs, training)
        return recon + z_entropy + y_entropy

    @tf.function
    def loss_fn(self, inputs, training=False):
        return -self.elbo(inputs, training)

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(x, training=True)
        # Update ops for batch normalization
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        gradients = tape.gradient(loss, self.trainable_variables)
        # Clipping
        gradients = [
            None if gradient is None else tf.clip_by_value(gradient, -1e-0, 1e0)
            for gradient in gradients
        ]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    @tf.function
    def predict(self, x, training=False):
        qy_g_x__logit, qy_g_x__prob = self.graph_qy_g_x(x, training)
        return qy_g_x__prob


@tf.function
def train(model, X_train, y_train, X_test, y_test, num, epochs, iter, verbose=10):

    print(
        "{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "epoch",
            "loss",
            "likelih",
            "z-prior",
            "y-prior",
            "trAMI",
            "teAMI",
            "trPUR",
            "tePUR",
        )
    )

    for i in tqdm(range(epochs), position=0):
        for j in tqdm(range(iter), position=1):
            idx_train = np.random.choice(X_train.shape[0], num)
            model.train_step(X_train[idx_train])

        if i % verbose == 0:
            # Evaluate training metrics
            recon, z_ent, y_ent = chain_call(model.entropy_fn, X_train, num)

            recon = np.array(recon).mean()
            z_ent = np.array(z_ent).mean()
            y_ent = np.array(y_ent).mean()

            loss = -(recon + z_ent + y_ent)

            idx_tr = chain_call(model.predict, X_train, num).argmax(1)
            idx_te = chain_call(model.predict, X_test, num).argmax(1)

            ami_tr = adjusted_mutual_info_score(y_train, idx_tr)
            ami_te = adjusted_mutual_info_score(y_test, idx_te)

            purity_train = purity_score(y_train, idx_tr)
            purity_test = purity_score(y_test, idx_te)

            print(
                "{:10d} {:10.5f} {:10.5f} {:10.5f} {:10.5f} "
                "{:10.5f} {:10.5f} "
                "{:10.5f} {:10.5f}".format(
                    i,
                    loss,
                    recon,
                    z_ent,
                    y_ent,
                    ami_tr,
                    ami_te,
                    purity_train,
                    purity_test,
                )
            )

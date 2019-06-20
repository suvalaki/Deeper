import tensorflow as tf 
import tensorflow_probability as tfp 

tfd = tfp.distributions
tfk = tf.keras

Model = tfk.Model

tf.enable_eager_execution()

import numpy as np


class Encoder(Model):

    def __init__(self, latent_dim, embedding_dimensions):
        Model.__init__(self,)
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
                    bias_initializer=tfk.initializers.Zeros()
                )
            )
            self.embeddings_bn.append(
                tf.layers.BatchNormalization()
            )

        self.latent = tfk.layers.Dense(
            units=self.latent_dim,
            activation=None,
            use_bias=False,
            kernel_initializer=tfk.initializers.he_normal(seed=None),
            bias_initializer=tfk.initializers.Zeros()
        )

    def call(self, inputs, training=False):
        """Define the computational flow"""
        x = inputs
        for em, bn in zip(self.embeddings, self.embeddings_bn):
            x = em(x)
            x = bn(x, training=training)
        x = self.latent(x)
        return x


class SoftmaxEncoder(Model):
    def __init__(self, latent_dim, embedding_dimensions):
        Model.__init__(self)
        self.latent_dim = latent_dim
        self.embedding_dimensions = embedding_dimensions

        self.logits = Encoder(self.latent_dim, self.embedding_dimensions)
            
    def call(self, inputs, training=False):
        x = inputs
        logits = self.logits(x)
        prob = tf.nn.softmax(logits)
        return logits, prob


class NormalEncoder(Model):
    def __init__(self, latent_dim, embedding_dimensions):
        Model.__init__(self)
        self.latent_dim = latent_dim
        self.embedding_dimensions = embedding_dimensions

        self.mu = Encoder(self.latent_dim, self.embedding_dimensions)
        self.logvar = Encoder(self.latent_dim, self.embedding_dimensions)
        self.sample = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfd.MultivariateNormalDiag(
                t[0], tf.exp(t[1])),
            convert_to_tensor_fn=lambda s: s.sample(1))
        
            
    def call(self, inputs, training=False):
        x = inputs
        mu = self.mu(x, training)
        logvar = self.logvar(x, training)
        dist = self.sample((mu, logvar))
        sample = dist.sample(1)
        
        # Metrics for loss
        logprob = dist.log_prob(sample)
        prob = dist.prob(sample)
        
        return sample, logprob, prob


class NormalDecoder(Model):
    def __init__(self, latent_dim, embedding_dimensions):
        Model.__init__(self)
        self.latent_dim = latent_dim
        self.embedding_dimensions = embedding_dimensions

        self.mu = Encoder(self.latent_dim, self.embedding_dimensions)
        self.logvar = Encoder(self.latent_dim, self.embedding_dimensions)
        self.sample = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfd.MultivariateNormalDiag(
                t[0], tf.exp(t[1])),
            convert_to_tensor_fn=lambda s: s.sample())
        
            
    def call(self, inputs, outputs, training=False, var=None):
        x = inputs
        mu = self.mu.call(x, training)
        #logvar = self.logvar(x, training)
        if var is not None:
            logvar = tf.exp(tf.cast(self.logvar(x, training), tf.float64))
        else:
            logvar = tf.fill(tf.shape(outputs), tf.cast(1.,tf.float64))
        dist = self.sample(
            (
                tf.cast(outputs, tf.float64),
                logvar
            )
        )
        #sample = dist.sample(1)
        # Metrics for loss
        #import pdb; pdb.set_trace()
        logprob = dist.log_prob(tf.cast(mu[:,:],tf.float64))
        prob = dist.prob(tf.cast(mu[:,:],tf.float64))
        
        return mu[None,:,:], logprob, prob


class SigmoidDecoder(Model):
    def __init__(self, latent_dim, embedding_dimensions):
        Model.__init__(self)
        self.latent_dim = latent_dim
        self.embedding_dimensions = embedding_dimensions
        self.mu = Encoder(self.latent_dim, self.embedding_dimensions)
        self.sample = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Bernoulli(logits=t),
            convert_to_tensor_fn=lambda s: s.sample())
        
    def call(self, inputs, outputs, training=False):
        x = inputs
        logit = self.mu.call(x, training)
        #logvar = self.logvar(x, training)
        #dist = self.sample(tf.cast(outputs, tf.float64),)
        #sample = dist.sample(1)
        # Metrics for loss
        #import pdb; pdb.set_trace()
        #logprob = dist.log_prob(tf.cast(mu[:,:],tf.float64))
        #prob = dist.prob(tf.cast(mu[:,:],tf.float64))
        eps = 0.01
        if eps > 0.0:
            max_val = np.log(1.0 - eps) - np.log(eps)
            logit = tf.clip_by_value(
                logit, -max_val, max_val
            )


        logprob = - tf.cast(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=outputs[None,:,:]),
            #outputs[None,:,:] * tf.log(tf.nn.sigmoid(logit)),
            -1), tf.float64)

        prob = tf.exp(logprob)

        return logit[None,:,:], logprob, prob



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
    ):

        #instatiate
        Model.__init__(self,)

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

        self.graph_qy_g_x = SoftmaxEncoder(self.k, self.em_dim)
        self.graphs_qz_g_xy = [
            NormalEncoder(self.la_dim, self.em_dim) 
            for i in range(self.k)
        ]
        self.graphs_pz_g_y = [
            NormalDecoder(self.la_dim, [int(self.la_dim//2)]) 
            for i in range(self.k)
        ]
        if self.kind == 'binary':
            self.graphs_px_g_zy = [
                SigmoidDecoder(self.in_dim, self.em_dim[::-1]) 
                for i in range(self.k)
            ]
        else:
            self.graphs_px_g_zy = [
                NormalDecoder(self.in_dim, self.em_dim[::-1]) 
                for i in range(self.k)
            ]
        

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

    @staticmethod
    def mc_stack_mean(x):
        return tf.identity(tf.stack(x, 0) / len(x))
        
    def call(self, inputs, training=False):
        x = inputs
                
        y_ = tf.fill(tf.stack([tf.shape(x)[0], self.k]), 0.0)
        py = tf.fill(tf.shape(y_), 1 / self.k, name="prob")
                
        qy_g_x__logit, qy_g_x__prob = self.graph_qy_g_x(x, training)
        
        qz_g_xy__sample = [[None]*self.mc_sam for i in range(self.k)]
        qz_g_xy__logprob = [[None]*self.mc_sam for i in range(self.k)]
        qz_g_xy__prob = [[None]*self.mc_sam for i in range(self.k)]
        
        pz_g_y__sample = [[None]*self.mc_sam for i in range(self.k)]
        pz_g_y__logprob = [[None]*self.mc_sam for i in range(self.k)]
        pz_g_y__prob = [[None]*self.mc_sam for i in range(self.k)]
        
        px_g_zy__sample = [[None]*self.mc_sam for i in range(self.k)]
        px_g_zy__logprob = [[None]*self.mc_sam for i in range(self.k)]
        px_g_zy__prob = [[None]*self.mc_sam for i in range(self.k)]
        
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
            for j in range(self.mc_sam):
                y = tf.add(
                    y_,
                    tf.constant(
                        np.eye(self.k)[i],
                        dtype=tf.float32,
                        name="y_one_hot".format(i),
                    ),
                )

                xy = tf.concat([x,y], axis=-1)
                (
                    qz_g_xy__sample[i][j], 
                    qz_g_xy__logprob[i][j], 
                    qz_g_xy__prob[i][j]
                 ) = self.graphs_qz_g_xy[i].call(xy, training)
                (
                    pz_g_y__sample[i][j], 
                    pz_g_y__logprob[i][j], 
                    pz_g_y__prob[i][j] 
                ) = self.graphs_pz_g_y[i].call(y, qz_g_xy__sample[i][j], training, var=True)

                (
                    px_g_zy__sample[i][j], 
                    px_g_zy__logprob[i][j], 
                    px_g_zy__prob[i][j]
                ) = self.graphs_px_g_zy[i].call(
                    qz_g_xy__sample[i][j], inputs, training)

            # Monte Carlo
            mc_qz_g_xy__sample[i] = self.mc_stack_mean(qz_g_xy__sample[i])
            #import pdb; pdb.set_trace()
            mc_qz_g_xy__logprob[i] = self.mc_stack_mean(qz_g_xy__logprob[i])
            mc_qz_g_xy__prob[i] = self.mc_stack_mean(qz_g_xy__prob[i])

            mc_pz_g_y__sample[i] = self.mc_stack_mean(pz_g_y__sample[i])
            mc_pz_g_y__logprob[i] = self.mc_stack_mean(pz_g_y__logprob[i])
            mc_pz_g_y__prob[i] = self.mc_stack_mean(pz_g_y__prob[i])

            mc_px_g_zy__sample[i] = self.mc_stack_mean(px_g_zy__sample[i])
            mc_px_g_zy__logprob[i] = self.mc_stack_mean(px_g_zy__logprob[i])
            mc_px_g_zy__prob[i] = self.mc_stack_mean(px_g_zy__prob[i])

        return (
            py, qy_g_x__prob,
            mc_qz_g_xy__sample, mc_qz_g_xy__logprob, mc_qz_g_xy__prob, 
            mc_pz_g_y__sample, mc_pz_g_y__logprob, mc_pz_g_y__prob,
            mc_px_g_zy__sample, mc_px_g_zy__logprob, mc_px_g_zy__prob
        )
            
    def entropy_fn(self, inputs, training=False):
        (
            py, qy_g_x,
            mc_qz_g_xy__sample, 
            mc_qz_g_xy__logprob, 
            mc_qz_g_xy__prob, 
            mc_pz_g_y__sample, 
            mc_pz_g_y__logprob, 
            mc_pz_g_y__prob,
            mc_px_g_zy__sample, 
            mc_px_g_zy__logprob, 
            mc_px_g_zy__prob
        ) = self.call(inputs, training=training)
        
        #reconstruction
        recon = tf.add_n([ tf.cast(qy_g_x[:,i],tf.float64) * tf.cast(mc_px_g_zy__logprob[i],tf.float64) for i in range(self.k) ])
        
        #z_entropy
        z_entropy = tf.add_n([ tf.cast(qy_g_x[:,i], tf.float64) * (
            tf.cast(mc_pz_g_y__logprob[i], tf.float64) 
            - tf.cast(mc_qz_g_xy__logprob[i], tf.float64)) for i in range(self.k) ])
        
        #y_entropy
        y_entropy = tf.reduce_sum(tf.cast(qy_g_x, tf.float64) * (
            tf.log(tf.cast(py,tf.float64)) - tf.log(tf.cast(qy_g_x, tf.float64))), axis=-1)
        
        #elbo = recon + z_entropy + y_entropy
        return recon, z_entropy, y_entropy

    def elbo(self, inputs, training=False):
        recon, z_entropy, y_entropy = self.entropy_fn(inputs, training)
        return recon + z_entropy + y_entropy

    def loss_fn(self,inputs, training=False):
        return - self.elbo(inputs, training)

    #@tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(x, training=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients = tape.gradient(loss, self.trainable_variables)
            # Clipping
            gradients = [
                None if gradient is None 
                else tf.clip_by_value(gradient,-1e-0,1e0)
                for gradient in gradients
            ]
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


    def predict(self, x, training=False):
        qy_g_x__logit, qy_g_x__prob = self.graph_qy_g_x(x, training)
        return qy_g_x__prob

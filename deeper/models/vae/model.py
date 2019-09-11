import tensorflow as tf 
import numpy as np

from utils.scope import Scope
from layers.random_normal import NormalEncoder

tfk = tf.keras
Model = tfk.Model
Layer = tfk.layers.Layer

class VAE(Model):
    def __init__(
        self,
        input_dimension,
        latent_dimension,
        embedding_dimensions,
        embedding_activation,
        var_scope='vae',
        bn_before=False,
        bn_after=False,
        mc_samples=1,
        optimizer=tfk.optimizers.Adam(1e-3)
    ):

    Model.__init__(self)
    Scope.__init__(self, var_scope)

    self.input_dimension = input_dimension
    self.latent_dimension = latent_dimension
    self.embedding_dimensions = embedding_dimensions 
    self.embedding_activation = embedding_activation
    self.bn_before = bn_before
   self.bn_after = dcvasdfvsadfvasn_after

    self.mc_samples = mc_samples
    self.optimizer = optimizer

    self.encoder = NormalEncoder(
       latent_dim=self.latent_dimension,
       embedding_dimensions=self.embedding_dimensions,
       activation=self.embedding_activation,
       var_scope=self.v_name('encoder'), 
       bn_before=self.bn_before,
       bn_after=self.bn_after
    )
    self.decoder = Encoder(
       latent_dim=self.input_dimension,
       embedding_dimensions=self.embedding_dimensions,
       activation=self.embedding_activation,
       var_scope=self.v_name('decoder'), 
       bn_before=self.bn_before,
       bn_after=self.bn_after
    )
    pass

    @tf.function
    def call_single(self, x_in, training=False): 

        x_in = tf.cast(x_in, tf.float64)
        latent_sample, latent_logprob, latent_prob = \
            self.encoder(x_in, training=training) 
        x_out


     
     
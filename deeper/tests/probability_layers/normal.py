import tensorflow as tf
import numpy as np 

from deeper.probability_layers.normal import RandomNormalEncoder


encoder = RandomNormalEncoder(
        latent_dimension=10, 
        embedding_dimensions=[10,10,], 
        embedding_activations=tf.nn.relu,
        var_scope='normal_encoder',
        bn_before=False,
        bn_after=False
)


x = np.array([np.random.uniform(0,1,10) for i in range(10)]).astype(float)

encoder(x)
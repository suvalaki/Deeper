import tensorflow as tf
import numpy as np

from deeper.models.gmvae.gmvae_pure_sampling.model import (
    MarginalAutoEncoder,
    Gmvae,
)

tf.enable_eager_execution()

x = np.array([np.random.normal(0, 1, 10) for i in range(15)])
y = np.array([np.random.uniform(0, 1, 1) for i in range(15)])

if False:
    auto_encoder = MarginalAutoEncoder(
        input_dimension=10,
        embedding_dimensions=[10, 10],
        embedding_activations=tf.nn.tanh,
        latent_dim=5,
        kind="binary",
        var_scope="marginal_autoencoder",
        bn_before=False,
        bn_after=False,
    )

    z0 = auto_encoder(x, y)

    for k in z0:
        print(k.shape)


gmvae = Gmvae(
    components=2,
    input_dimension=10,
    embedding_dimensions=[10, 10],
    latent_dimensions=3,
    embedding_activations=tf.nn.relu,
    mixture_embedding_activations=None,
    mixture_embedding_dimensions=None,
    bn_before=False,
    bn_after=False,
    categorical_epsilon=0.0,
    latent_epsilon=0.0,
    reconstruction_epsilon=1e-8,
    kind="binary",
    learning_rate=0.001,
    gradient_clip=None,
)

qy = gmvae.graph_qy_g_x(x, training=False)
gmvae.graph_qy_g_x_ohe(qy[1], 1.0)

gmvae.sample_one(x, False, 1)

z = gmvae(x)

gmvae.elbo(x)

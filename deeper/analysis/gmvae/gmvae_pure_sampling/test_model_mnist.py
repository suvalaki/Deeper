#%%
import tensorflow as tf
import numpy as np
from deeper.gmvae.gmvae_pure_sampling import model 
from tensorflow.examples.tutorials.mnist import input_data

from deeper.gmvae.gmvae_pure_sampling.utils import chain_call, purity_score
from deeper.gmvae.gmvae_pure_sampling.train import train

#%%
# Import One Hot Encoded MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X_train = mnist.train.images
y_train_arr = mnist.train.labels
y_train = y_train_arr.argmax(1)
X_test = mnist.test.images
y_test_arr = mnist.test.labels
y_test = y_test_arr.argmax(1)

#%%
# Instantiate the model

m1 = model.Gmvae(
    components=len(set(y_train)),
    input_dimension=X_train.shape[1],
    embedding_dimensions=[512,512],
    latent_dimensions=256,
    kind="binary",
    monte_carlo_samples=5,
    learning_rate=0.001
)

#%%
# Run the training opperation
for i in range(10):
    idx_train = np.random.choice(X_train.shape[0], 100)
    m1.train_step(X_train[idx_train])

#%%
idx_train = np.random.choice(X_train.shape[0], 10)

recon, z_ent, y_ent = m1.entropy_fn(X_train[idx_train])

recon = np.array(recon).mean()
z_ent = np.array(z_ent).mean()
y_ent = np.array(y_ent).mean()

#%%
recon

#%%
z_ent

#%%
y_ent

#%%
train(m1, X_train, y_train, X_test, y_test, num=500, epochs=50, iter=100)

#%%

#%%
from pathlib import Path
import tensorflow as tf
import numpy as np
from deeper.models.gmvae.gmvae_marginalised_categorical import model
from deeper.models.gmvae.gmvae_marginalised_categorical.utils import (
    chain_call, chain_call_dataset, purity_score )

print('tensorflow gpu available {}'.format(tf.test.is_gpu_available()))

#%% Checlk whether the log directory exists. If it does not create it and empty
logfolder = Path('./logs/test_model/')
#if logpathis_dir():
    


#%% Load MNIST and make it binary encoded
mnist = tf.keras.datasets.mnist
(X_train, y_train),(X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape(X_train.shape[0],28*28)
X_test = X_test.reshape(X_test.shape[0],28*28)
X_train = (X_train > 0.5).astype(float)
X_test = (X_test > 0.5).astype(float)

#%% Instantiate the model
from importlib import reload
model = reload(model)
with tf.device('/gpu:0'):
    m1 = model.Gmvae(
        components=len(set(y_train)),
        input_dimension=X_train.shape[1],
        embedding_dimensions=[512,512],
        latent_dimensions=256,
        kind="binary",
        monte_carlo_samples=10,
        learning_rate=0.01
    )

#m1.compile(loss=m1.loss_fn, optimizer=m1.optimizer)

#m1.fit(X_train, 10)

#%% Initialize the Graph by running th training op once
idx_train = np.random.choice(X_train.shape[0], 100)
recon, z_ent, y_ent = m1.entropy_fn(X_train[idx_train])

recon = np.array(recon).mean()
z_ent = np.array(z_ent).mean()
y_ent = np.array(y_ent).mean()

print('recon: {}\nz_ent: {}\ny_ent: {}'.format(recon,z_ent,y_ent))

#%% Train the model
from deeper.models.gmvae.gmvae_marginalised_categorical.train import train
#with tf.device('/gpu:0'):
train(
    m1, 
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    num=200, 
    epochs=10, 
    iter=1, 
    verbose=1
)

#%% check gpu on training cycle
num=1000
epochs = 1
iter = 100
for i in tqdm(range(epochs), position=0):
    for j in tqdm(range(iter), position=1):
        idx_train = np.random.choice(X_train.shape[0],num)
        m1.train_step(X_train[idx_train])


while True:
    idx_train = np.random.choice(X_train.shape[0],num)
    #m1.train_step(X_train[idx_train])

    h=m1.entropy_fn(X_train[idx_train])

#%% Check the train call for the bottlekneck
recon, z_ent, y_ent = chain_call(m1.entropy_fn, X_train, num)
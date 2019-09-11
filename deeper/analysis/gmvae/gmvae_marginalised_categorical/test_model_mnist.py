#%%
from pathlib import Path
import tensorflow as tf

tf.enable_eager_execution()

import numpy as np
from deeper.models.gmvae.gmvae_marginalised_categorical import model
from deeper.models.gmvae.gmvae_marginalised_categorical.utils import (
    chain_call,
    chain_call_dataset,
    purity_score,
)

print("tensorflow gpu available {}".format(tf.test.is_gpu_available()))

#%% Checlk whether the log directory exists. If it does not create it and empty
logfolder = Path("./logs/test_model/")
# if logpathis_dir():

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            #tf.config.experimental.gpu.set_per_process_memory_fraction(0.9)
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_synchronous_execution(True)
            #tf.config.experimental.set_per_process_memory_fraction( 0.9)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#%% Load MNIST and make it binary encoded
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape(X_train.shape[0], 28 * 28)
X_test = X_test.reshape(X_test.shape[0], 28 * 28)
X_train = (X_train > 0.5).astype(float)
X_test = (X_test > 0.5).astype(float)

#%% Instantiate the model
from importlib import reload

model = reload(model)

m1 = model.Gmvae(
    components=len(set(y_train)),
    input_dimension=X_train.shape[1],
    embedding_dimensions=[512, 512],
    latent_dimensions=64,
    kind="binary",
    monte_carlo_samples=1,
    learning_rate=1e-3,
    gradient_clip=10000
)

from deeper.models.gmvae.gmvae_marginalised_categorical.train import train

#%% Train the model
# with tf.device('/gpu:0'):
train(
    m1, 
    X_train, y_train, 
    X_test, y_test, 
    num=10, 
    samples=1,
    epochs=10000, 
    iter_train=1, 
    num_inference=1000, 
    save='model_w'
)


#%%

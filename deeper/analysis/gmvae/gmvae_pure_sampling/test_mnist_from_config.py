#%%
from deeper.models.gmvae.gmvae_pure_sampling import wrapper
import json
import numpy as np
import tensorflow as tf

CONFIG_FILE = "./configs/models/gmvae/gmvae_pure_sampling/defaultconfig.json"
CONFIG_TRAIN_FILE = "./configs/models/gmvae/gmvae_pure_sampling/defaulttrainconfig.json"

tf.keras.backend.set_floatx("float64")

#%% Load MNIST and make it binary encoded
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape(X_train.shape[0], 28 * 28)
X_test = X_test.reshape(X_test.shape[0], 28 * 28)
X_train = (X_train > 0.5).astype(float)
X_test = (X_test > 0.5).astype(float)


#%% Create Model from config file
config_dict = json.loads(open(CONFIG_FILE).read())
model = wrapper.ModelWrapper(config_dict)

#%% Train the model from configuration file
train_config_dict = json.loads(open(CONFIG_TRAIN_FILE).read())
model.train_from_config(X_train, y_train, X_test, y_test, train_config_dict)


# %%

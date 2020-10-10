#%%
from pathlib import Path
import tensorflow as tf
import numpy as np
from tqdm import tqdm

# tf.enable_v2_behavior()
# tf.enable_eager_execution()
tf.random.set_seed(123154)
# tf.keras.backend.set_floatx('float64')

import numpy as np
from deeper.models.vae.model import VAE as model
from deeper.models.vae.utils import chain_call
from deeper.utils.metrics import purity_score
from deeper.models.vae.train import train
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder
from deeper.utils.metrics import purity_score
import deeper.utils.cooling as cooling

print("tensorflow gpu available {}".format(tf.test.is_gpu_available()))

#%% Checlk whether the log directory exists. If it does not create it and empty
logfolder = Path("./logs/test_model/")
# if logpathis_dir():

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            # tf.config.experimental.gpu.set_per_process_memory_fraction(0.9)
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_synchronous_execution(True)
            # tf.config.experimental.set_per_process_memory_fraction( 0.9)
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

ohe = OneHotEncoder()
y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1)).astype(float)
y_test_ohe = ohe.transform(y_test.reshape(-1, 1)).astype(float)

#%% Instantiate the model
params = {
    "input_regression_dimension": 0,
    "input_boolean_dimension": X_train.shape[1],
    "input_categorical_dimension": 0,
    "output_regression_dimension": 0,
    "output_boolean_dimension": X_train.shape[1],
    "output_categorical_dimension": 10,
    "encoder_embedding_dimensions": [512, 512],
    "decoder_embedding_dimensions": [512, 512],
    "latent_dim": 64,
    "embedding_activations": tf.nn.relu,
    "kind": "binary",
    "bn_before": False,
    "bn_after": True,
    "reconstruction_epsilon": 1e-12,
    "latent_epsilon": 1e-12,
    "optimizer": tf.keras.optimizers.Adam(1e-3, epsilon=1e-16),
    "connected_weights": False,
    "latent_mu_embedding_dropout": 0.0,
    "latent_var_embedding_dropout": 0.0,
    "recon_dropouut": 0.0,
    #'latent_fixed_var': 10.0,
}

param_string = "vae" + "__".join(
    [str(k) + "_" + str(v) for k, v in params.items()]
)

m1 = model(**params)

# %%
val = np.concatenate([X_train, y_train_ohe.todense()], 1)[0:10]
samp = m1.sample_one(X_train[0:10], val)

#%% Prediction

pred = m1.predict_one(X_train[0:10])

# %%

z_cooling = lambda: 1.0

#%% Train the model
# with tf.device('/gpu:0'):
train(
    m1,
    X_train,
    X_test,
    np.concatenate([X_train, y_train_ohe.todense()], 1),
    np.concatenate([X_test, y_test_ohe.todense()], 1),
    num=100,
    samples=1,
    epochs=1500,
    iter_train=1,
    num_inference=1000,
    save="model_w_5",
    batch=True,
    save_results="./gumble_results.txt",
    beta_z_method=z_cooling,
    tensorboard=None,  # "./logs/" + param_string + "/samples__" + str(1),
)

# %%

res_tensors = m1(X_test)

#%% Display an original image
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt

k = 7


# plt.subplot()
# imshow(image, cmap='gray')

w = h = 28
fig, axes = plt.subplots(2, 10, figsize=(25, 5))
for j in range(10):
    image = X_test[k + j].reshape(w, h)
    # imshow(image, cmap='gray')

    # Display a predicted imgage
    image2 = res_tensors["x_recon_bin"][k + j].numpy().reshape(w, h)

    axes[0, j].imshow(image)
    axes[1, j].imshow(image2)


# %%

#%%
import sys

sys.path.append("../../../..")
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from pathlib import Path
import tensorflow as tf
import io

# USE CPU ONLY
# tf.config.set_visible_devices([], "GPU")
import numpy as np
from tqdm import tqdm
import json
import tensorflow_addons as tfa
from matplotlib import pyplot as plt

# tf.enable_eager_execution()

# tf.keras.backend.set_floatx("float64")

import numpy as np

from deeper.models.vae import Vae, MultipleObjectiveDimensions

from deeper.utils.cooling import exponential_multiplicative_cooling
import deeper.utils.cooling as cooling

from sklearn import metrics
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    adjusted_mutual_info_score,
)
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder

from deeper.models.gan.model import Gan
from deeper.models.gan.descriminator import DescriminatorNet

from deeper.analysis.gan.callbacks import PlotterCallback


print("tensorflow gpu available {}".format(tf.test.is_gpu_available()))


#%% Load MNIST and make it binary encoded
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

(X_train_og, y_train_og), (X_test_og, y_test_og) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28 * 28)
X_test = X_test.reshape(X_test.shape[0], 28 * 28)
X_train = (X_train > 0.5).astype(float)
X_test = (X_test > 0.5).astype(float)

#%% Filter to a single label
if False:
    LABEL = 9
    X_train_og = X_train_og[y_train == LABEL]
    X_test_og = X_test_og[y_test == LABEL]

    X_train = X_train[y_train == LABEL]
    X_test = X_test[y_test == LABEL]


#%% Instantiate the model
BATCH_SIZE = 12
desciminatorConfig = DescriminatorNet.Config(
    embedding_dimensions=[512, 512, 128],
    activation=tf.keras.layers.Activation("relu"),
    embedding_dropout=0.25,
    # bn_before=True,
)
vaeConfig = Vae.Config(
    input_dimensions=MultipleObjectiveDimensions(
        regression=0,
        boolean=X_train.shape[-1],
        ordinal=(0,),
        categorical=(0,),
    ),
    output_dimensions=MultipleObjectiveDimensions(
        regression=0,
        boolean=X_train.shape[-1],
        ordinal=(0,),
        categorical=(0,),
    ),
    encoder_embedding_dimensions=[512, 512, 256],
    decoder_embedding_dimensions=[512, 512, 256][::-1],
    latent_dim=64,
    embedding_activations=tf.keras.layers.ELU(),
    # bn_before=True,
)


config = Gan.Config(descriminator=desciminatorConfig, generator=vaeConfig, training_ratio=5)

model = Gan(config)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.000005))
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
# model.compile(optimizer=tf.keras.optimizers.SGD(1e-3))
# model.compile()

#%% train
fp = "./logs/gan/test_mnist_gumblegmvae"
model.fit(
    X_train,
    X_train,
    callbacks=[
        tf.keras.callbacks.TensorBoard(fp),
        PlotterCallback(fp, X_train, X_test, y_train, y_test, model),
    ],
    epochs=10000,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, X_test),
)

# %% Real
plt.imshow(X_train_og[0], cmap="gray")
# %% Fake
y_pred_train = model(X_train[0:1])
plt.imshow(y_pred_train.numpy().reshape((28, 28)), cmap="gray")
# %%
# By the nature of the GAN just trying to learn how to fool the descriminator
# we have no garuantee that the input digit will look like the output digit.
# The output could be any of the digits which fools the descriminator.

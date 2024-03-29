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

from deeper.analysis.generalised_autoencoder.callbacks import (
    ReconstructionImagePlotter,
    ClusteringCallback,
    LatentPlotterCallback,
)


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
ds_train = tf.data.Dataset.from_tensor_slices((X_train, X_train)).batch(BATCH_SIZE)
ds_test = tf.data.Dataset.from_tensor_slices((X_test, X_test)).batch(BATCH_SIZE)

#%% Instantiate the model
def build_model(hp):

    desciminatorConfig = DescriminatorNet.Config(
        embedding_dimensions=[512, 512, 128],
        activation=tf.keras.layers.Activation("relu"),
        embedding_dropout=0.25,
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
        embedding_activations=tf.keras.layers.Activation("relu"),
    )
    config = Gan.Config(
        descriminator=desciminatorConfig,
        generator=vaeConfig,
        training_ratio=1,
    )
    config = config.parse_tunable(hp)

    model = Gan(config)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.000005))
    return model


#%%
import keras_tuner as kt

tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_losses/loss", direction="min"),
    # max_trials=150,
    max_epochs=1000,
)
tuner._display.col_width = 60
tuner.search(ds_train, epochs=1, validation_data=ds_test)
tuner.search(ds_train, epochs=1, validation_data=ds_test)
best_model = tuner.get_best_models()[0]

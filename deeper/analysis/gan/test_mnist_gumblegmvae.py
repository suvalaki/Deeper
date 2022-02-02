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
from deeper.models.gmvae.gmvae_pure_sampling import GumbleGmvae

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
desciminatorConfig = DescriminatorNet.Config(
    embedding_dimensions=[512, 512, 128],
    activation=tf.keras.layers.Activation("relu"),
    embedding_dropout=0.25,
    # bn_before=True,
)
vaeConfig = GumbleGmvae.Config(
    components=10,
    cat_embedding_dimensions=[512, 512, 256],
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
    embedding_activation=tf.keras.layers.ELU(),
    gumble_temperature_schedule=tfa.optimizers.CyclicalLearningRate(
        0.5,
        1.0,
        step_size=10000.0,
        scale_fn=lambda x: 1 / (1.2 ** (x - 1)),
        scale_mode="cycle",
    ),
    # gumble_temperature_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    #     boundaries=[
    #         X_train.shape[0] * 5 // BATCH_SIZE ,
    #         X_train.shape[0] * 10 // BATCH_SIZE,
    #         X_train.shape[0] * 20 // BATCH_SIZE
    #     ],
    #     values=[5.0, 1.0, 0.75, 0.5],
    # ),
    kld_y_schedule=tfa.optimizers.CyclicalLearningRate(
        1.0, 1.0, step_size=30000.0, scale_fn=lambda x: 1.0, scale_mode="cycle"
    ),
    kld_z_schedule=tfa.optimizers.CyclicalLearningRate(
        1.0, 1.0, step_size=30000.0, scale_fn=lambda x: 1.0, scale_mode="cycle"
    ),
    # bn_before=True
)


config = Gan.Config(descriminator=desciminatorConfig, generator=vaeConfig, training_ratio=5)

model = Gan(config)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.000005))

#%% train
fp = "./logs/gan/test_mnist_gumblegmvae"
tbc = tf.keras.callbacks.TensorBoard(fp)
rc = ReconstructionImagePlotter(model, tbc, X_train, X_test, y_train, y_test)
cc = ClusteringCallback(model, tbc, X_train, X_test, y_train, y_test)
lc = LatentPlotterCallback(model, tbc, X_train, X_test, y_train, y_test)
model.fit(
    X_train,
    X_train,
    callbacks=[tbc, rc, cc, lc],
    epochs=10000,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, X_test),
)

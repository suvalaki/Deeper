#%%
import sys

sys.path.append("../../../..")
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from pathlib import Path
import tensorflow as tf

# USE CPU ONLY
tf.config.set_visible_devices([], "GPU")
import numpy as np
from tqdm import tqdm
import json
import tensorflow_addons as tfa

# tf.enable_eager_execution()

# tf.keras.backend.set_floatx("float64")

import numpy as np

from deeper.models.gmvae import MultipleObjectiveDimensions
from deeper.models.gmvae.gmvae_pure_sampling import GumbleGmvae

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
X_train = X_train.reshape(X_train.shape[0], 28 * 28)
X_test = X_test.reshape(X_test.shape[0], 28 * 28)
X_train = (X_train > 0.5).astype(float)
X_test = (X_test > 0.5).astype(float)

# y_ohe = OneHotEncoder()
# y_train_ohe = np.array(y_ohe.fit_transform(y_train.reshape(-1,1)).todense())
# y_test_ohe = np.array(y_ohe.transform(y_test.reshape(-1,1)).todense())


#%% Instantiate the model
BATCH_SIZE = 64
ds_train = (
    tf.data.Dataset.from_tensor_slices((X_train, X_train))
    .shuffle(X_train.shape[0], reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
)
config = GumbleGmvae.Config(
    monte_carlo_training_samples=5,
    components=10,
    cat_embedding_dimensions=[512, 512],
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
    encoder_embedding_dimensions=[512, 512, 128],
    decoder_embedding_dimensions=[512, 512, 128][::-1],
    latent_dim=64,
    embedding_activation=tf.keras.layers.ReLU(),
    gumble_temperature_schedule=tf.keras.optimizers.schedules.PolynomialDecay(
        1.0,
        X_train.shape[0] * 50 // BATCH_SIZE,
        end_learning_rate=0.5,
        power=1.0,
        cycle=False,
        name=None,
    ),
    kld_y_schedule=tfa.optimizers.CyclicalLearningRate(
        1.0, 1.0, step_size=30000.0, scale_fn=lambda x: 1.0, scale_mode="cycle"
    ),
    kld_z_schedule=tfa.optimizers.CyclicalLearningRate(
        2.0, 2.0, step_size=30000.0, scale_fn=lambda x: 1.0, scale_mode="cycle"
    ),
    bn_before=True,
)

model = GumbleGmvae(config, dtype=tf.float64)
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(
#         tf.keras.optimizers.schedules.ExponentialDecay(
#             1e-3,
#             X_train.shape[0] * 1 // BATCH_SIZE,
#             0.5,
#             staircase=False,
#             name=None,
#         )
#     )
# )
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

#%% train
#!rm ./logs/gmvae/gumble_22 -rf
fp = "./logs/gmvae/gumble_27_sgd_replication0"
tbc = tf.keras.callbacks.TensorBoard(fp)
rc = ReconstructionImagePlotter(model, tbc, X_train, X_test, y_train, y_test)
cc = ClusteringCallback(model, tbc, X_train, X_test, y_train, y_test)
lc = LatentPlotterCallback(model, tbc, X_train, X_test, y_train, y_test)

model.fit(
    ds_train,
    epochs=2000,
    callbacks=[tbc, rc, cc, lc],
    # batch_size=BATCH_SIZE,
    validation_data=(X_test, X_test),
)

#%% category

y_pred_train = model.predict(X_train)

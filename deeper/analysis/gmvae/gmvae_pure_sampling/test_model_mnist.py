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
from deeper.models.gmvae.metrics import PurityCallback

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
BATCH_SIZE = 24
config = GumbleGmvae.Config(
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

model = GumbleGmvae(config)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
# model.compile()

#%% train
tbc = tf.keras.callbacks.TensorBoard(
    "./logs/trial_4_fix_z_sched_z_schedule_only"
)
pc = PurityCallback(tbc, X_train, X_test, y_train, y_test)
model.fit(
    X_train,
    X_train,
    epochs=10000,
    callbacks=[tbc, pc],
    batch_size=BATCH_SIZE,
    validation_data=(X_test, X_test),
)

#%% category

y_pred_train = model.predict(X_train)

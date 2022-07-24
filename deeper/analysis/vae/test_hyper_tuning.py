#%%
import sys

sys.path.append("../../../..")
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from pathlib import Path
import tensorflow as tf

# tf.config.experimental_run_functions_eagerly(True)

# USE CPU ONLY
tf.config.set_visible_devices([], "GPU")
import numpy as np
from tqdm import tqdm
import json
import tensorflow_addons as tfa

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

from deeper.optimizers.automl.tunable_types import (
    TunableModelMixin,
    TunableActivation,
    OptionalTunableL1L2Regulariser,
    TunableL1L2Regulariser,
)

from deeper.models.vae.encoder import VaeEncoderNet
from deeper.models.vae.decoder import VaeReconstructionNet

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

BATCH_SIZE = 128
ds_train = tf.data.Dataset.from_tensor_slices((X_train, X_train)).batch(BATCH_SIZE)
ds_test = tf.data.Dataset.from_tensor_slices((X_test, X_test)).batch(BATCH_SIZE)


#%% Instantiate the model
def build_model(hp):

    config = Vae.Config(
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
        # encoder_embedding_dimensions=[512, 512, 256],
        # decoder_embedding_dimensions=[512, 512, 256][::-1],
        # latent_dim=64,
        embedding_activation=TunableActivation("elu"),
        # kld_z_schedule=tfa.optimizers.CyclicalLearningRate(
        #     1.0, 1.0, step_size=30000.0, scale_fn=lambda x: 1.0, scale_mode="cycle"
        # ),
        # bn_before=True
    )

    config = config.parse_tunable(hp)

    print(config)

    model = Vae(config)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), run_eagerly=False)
    return model


#%%
import keras_tuner as kt

tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_losses/loss", direction="min"),  # max_trials=150,
    max_epochs=1000,
)
#    def show_hyperparameter_table(self, trial):
tuner._display.col_width = 60
tuner.search(ds_train, epochs=1, validation_data=ds_test)
best_model = tuner.get_best_models()[0]

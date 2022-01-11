#%%
import sys 
sys.path.append("../../../..")
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from pathlib import Path
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import json
import tensorflow_addons as tfa

#tf.enable_eager_execution()

#tf.keras.backend.set_floatx("float64")

import numpy as np

from deeper.models.gmvae.gmvae_marginalised_categorical import Gmvae
from deeper.models.gmvae.metrics import PurityCallback

from deeper.utils.cooling import exponential_multiplicative_cooling
import deeper.utils.cooling as cooling

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
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
BATCH_SIZE = 32
ds_train = tf.data.Dataset.from_tensor_slices((X_train, X_train)).shuffle(X_train.shape[0], reshuffle_each_iteration=True).batch(BATCH_SIZE)
ds_test = tf.data.Dataset.from_tensor_slices((X_test, X_test))
config = Gmvae.Config(
    components = 10,
    input_regression_dimension = 0 ,
    input_boolean_dimension = X_train.shape[-1],
    input_ordinal_dimension = [0], 
    input_categorical_dimension = [0],
    output_regression_dimension= 0,
    output_boolean_dimension = X_train.shape[-1],
    output_ordinal_dimension = [0], 
    output_categorical_dimension = [0],
    cat_embedding_dimensions = [512, 512, 128],
    encoder_embedding_dimensions = [512, 512, 128],
    decoder_embedding_dimensions = [512, 512, 128][::-1],
    latent_dim = 64,
    embedding_activation=tf.keras.layers.ELU(),
    kld_y_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[
            X_train.shape[0] * 5 // BATCH_SIZE, 
            X_train.shape[0] * 20 // BATCH_SIZE
        ],
        values=[1.0, 1.0, 1.0],
    ),
    kld_z_schedule = tfa.optimizers.CyclicalLearningRate(
        1.0, 1.0, step_size=30000.0, scale_fn=lambda  x: 1 / (1.3 ** (x - 1)), scale_mode="cycle"
    ),
    #categorical_epsilon=1e-3
    #bn_before=True 
)

model = Gmvae(config, dtype=tf.dtypes.float64)
#model.compile(optimizer=tf.keras.optimizers.Adam())
model.compile()

#%% AccuracyCallback 
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(
        contingency_matrix
    )


#%% train
tbc = tf.keras.callbacks.TensorBoard("./logs/stacked_trial_3_fix_z_sched_z_schedule_only")
pc = PurityCallback(tbc, X_train, X_test, y_train, y_test)
model.fit(
    ds_train,
    epochs = 10000,
    callbacks=[tbc,  pc],
    batch_size=BATCH_SIZE,
    validation_data=(X_test, X_test)
)

#%% category 

y_pred_train = model.predict(X_train)


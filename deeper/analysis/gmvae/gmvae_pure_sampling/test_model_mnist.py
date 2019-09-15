#%%
from pathlib import Path
import tensorflow as tf
import numpy as np
from tqdm import tqdm

tf.enable_eager_execution()

import numpy as np
from deeper.models.gmvae.gmvae_pure_sampling import model
from deeper.models.gmvae.gmvae_pure_sampling.utils import (
    chain_call,
    chain_call_dataset,
    purity_score,
)
from deeper.models.gmvae.gmvae_pure_sampling.train import train
from deeper.utils.cooling import exponential_multiplicative_cooling

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder



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

initial_learning_rate = 1e-3
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.8,
    staircase=True)

m1 = model.Gmvae(
    components=len(set(y_train)),
    input_dimension=X_train.shape[1],
    embedding_dimensions=[512, 512],
    latent_dimensions=64,
    kind="binary",
    learning_rate=initial_learning_rate,
    gradient_clip=10000,
    bn_before=False
)



#%% Examine SOftmax Distribution
import pandas as pd

logits, prob = m1.graph_qy_g_x.call(X_test, training=False)

logit_df = pd.DataFrame()
for col in range(logits.shape[1]):
    temp_df = pd.DataFrame({'value':logits[:,col]})
    temp_df['k'] = str(col)
    logit_df = logit_df.append(temp_df, ignore_index=True)
import seaborn as sns
sns.violinplot(data=logit_df, y='value' , x='k')

# Plot the correct prediction densities



# confusion matrix of the classification
# for each caategory map the appropriate prediction
confusion_matrix(y_test, np.argmax(m1.predict(X_test),1))

#%% Train the model
# with tf.device('/gpu:0'):
train(
    m1, 
    X_train, y_train, 
    X_test, y_test, 
    num=10, 
    samples=10,
    epochs=10000, 
    iter_train=1, 
    num_inference=1000, 
    save='model_w',
    batch=False,
    temperature_function=lambda x: 5
)






#%% Train the model
# with tf.device('/gpu:0'):
train(
    m1, 
    X_train, y_train, 
    X_test, y_test, 
    num=10, 
    samples=10,
    epochs=10000, 
    iter_train=1, 
    num_inference=1000, 
    save='model_w',
    batch=False,
    temperature_function=lambda x: exponential_multiplicative_cooling(
        x, 5, 0.01, 0.96)
)


#%%o
qy_g_x__logit, qy_g_x__prob = m1.graph_qy_g_x(X_train[[0]])
m1.graph_qy_g_x_ohe(qy_g_x__prob, 50.0)

#%%

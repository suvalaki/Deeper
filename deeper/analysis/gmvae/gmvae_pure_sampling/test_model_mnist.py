#%%

import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from pathlib import Path
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import json

# tf.enable_eager_execution()

tf.keras.backend.set_floatx("float64")

import numpy as np
from deeper.models.gmvae.gmvae_pure_sampling import model
from deeper.models.gmvae.gmvae_pure_sampling.utils import (
    chain_call,
    chain_call_dataset,
    purity_score,
)
from deeper.models.gmvae.gmvae_pure_sampling.train import (
    train,
    train_even,
    pretrain_with_clusters,
)
from deeper.utils.cooling import exponential_multiplicative_cooling
import deeper.utils.cooling as cooling

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

# y_ohe = OneHotEncoder()
# y_train_ohe = np.array(y_ohe.fit_transform(y_train.reshape(-1,1)).todense())
# y_test_ohe = np.array(y_ohe.transform(y_test.reshape(-1,1)).todense())


#%% Instantiate the model
from importlib import reload

model = reload(model)


initial_learning_rate = 1e-2
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=5000, decay_rate=0.5, staircase=True
)

seed = 12345

tf.random.set_seed(seed)

params = {
    "components": len(set(y_train)),
    "input_dimension": X_train.shape[1],
    "embedding_dimensions": [512, 512],
    "latent_dimensions": 64,
    "mixture_embedding_dimensions": [512, 512],
    "mixture_latent_dimensions": 64,
    "embedding_activations": tf.nn.relu,
    "kind": "binary",
    "learning_rate": initial_learning_rate,
    "gradient_clip": None,
    "bn_before": True,
    "bn_after": False,
    "categorical_epsilon": 0.0,
    "reconstruction_epsilon": 0.0,
    "latent_epsilon": 0.0,
    "latent_prior_epsilon": 0.0,
    "z_kl_lambda": 1.0,
    "c_kl_lambda": 1.0,
    "cat_latent_bias_initializer": None,
    "connected_weights": True,
    # "optimizer":tf.keras.optimizers.Adam(lr_schedule, epsilon=1e-16),
    "optimizer": tf.keras.optimizers.Adam(1e-3, epsilon=1e-16),
    "categorical_latent_embedding_dropout": 0.0,
    "mixture_latent_mu_embedding_dropout": 0.0,
    "mixture_latent_var_embedding_dropout": 0.0,
    "mixture_posterior_mu_dropout": 0.0,
    "mixture_posterior_var_dropout": 0.0,
    "recon_dropouut": 0.0,
    #'latent_fixed_var': 0.01,
}

m1 = model.Gmvae(**params)

params["embedding_activations"] = "tanh"
params["optimizer"] = "adam_1e-3_1e-9"

param_string = (
    "/seed__"
    + str(seed)
    + "/"
    + "/".join([str(k) + "_" + str(v) for k, v in params.items()])
)

#%%
# m1.load_weights("model_w_5")

#%%
res = m1.call(X_test)


#%% Examine SOftmax Distribution
import pandas as pd

logits, prob = m1.graph_qy_g_x.call(X_test, training=False)

logit_df = pd.DataFrame()
for col in range(logits.shape[1]):
    temp_df = pd.DataFrame({"value": prob[:, col]})
    temp_df["k"] = str(col)
    logit_df = logit_df.append(temp_df, ignore_index=True)
import seaborn as sns

sns.violinplot(data=logit_df, y="value", x="k")

# Plot the correct prediction densities


# confusion matrix of the classification
# for each caategory map the appropriate prediction
confusion_matrix(y_test, np.argmax(m1.predict(X_test), 1))


#%% Pretrain the model with some known clusters
if False:
    pretrain_with_clusters(
        m1,
        X_train,
        y_train,
        X_test,
        y_test,
        num=100,
        samples=1,
        epochs=5,
        iter_train=1,
        num_inference=1000,
        save="model_w",
    )

#%%
if False:
    # Pretraining with even mixture losses
    train_even(
        m1,
        X_train,
        y_train,
        X_test,
        y_test,
        num=100,
        samples=1,
        epochs=10,
        iter_train=1,
        num_inference=1000,
        save="model_w_2",
        batch=True,
        temperature_function=lambda x: exponential_multiplicative_cooling(
            x, 0.5, 0.5, 0.99
        ),
        # temperature_function = lambda x: 0.1
        save_results="./gumble_results.txt",
    )


#%% setup cooling for trainign loop constants

# z_cooling = cooling.CyclicCoolingRegime(cooling.linear_cooling, 1e-1, 1, 25, 35)
# y_cooling = cooling.CyclicCoolingRegime(cooling.linear_cooling, 10.0, 1.0, 25, 35)

z_cooling = lambda: 1.0
y_cooling = lambda: 1.0


#%% Train the model
# with tf.device('/gpu:0'):
train(
    m1,
    X_train,
    y_train,
    X_test,
    y_test,
    num=100,
    samples=1,
    epochs=1500,
    iter_train=1,
    num_inference=1000,
    save="model_w_5",
    batch=True,
    temperature_function=lambda x: exponential_multiplicative_cooling(
        x, 1.0, 0.5, 0.99
    ),
    # temperature_function = lambda x: 0.1
    save_results="./gumble_results.txt",
    beta_z_method=z_cooling,
    beta_y_method=y_cooling,
    tensorboard="./logs/" + param_string + "/samples__" + str(1),
)


#%%
qy_g_x__logit, qy_g_x__prob = m1.graph_qy_g_x(X_train[[9]])
qy_g_x__ohe = np.array(
    [m1.graph_qy_g_x_ohe(qy_g_x__prob, 0.005).numpy()[0] for i in range(1000)]
)

#%%
logit_df = pd.DataFrame()
for col in range(qy_g_x__ohe.shape[1]):
    temp_df = pd.DataFrame({"value": qy_g_x__ohe[:, col]})
    temp_df["k"] = str(col)
    logit_df = logit_df.append(temp_df, ignore_index=True)
import seaborn as sns

sns.boxplot(data=logit_df, y="value", x="k")


#%%
for y in m1.entropy_fn(X_train[0:3]):
    print(tf.shape(y))

#%%

for y in m1.call(X_train[0:3]):
    print(tf.shape(y))

#%%
m1.call(X_train[0:3])

#%%
import matplotlib.pyplot as plt
import random

images_to_plot = 100

# Get the index for 10 from each predicted category
cats = np.argmax(m1.predict(X_test), 1)
cats_true = {}
idx = [i for i in range(len(cats))]
sample_idx = []
for col in set(cats):
    for i in range(10):
        sample_idx.append(np.random.choice(np.array(idx)[cats == col], 1))

    cats_true.update(
        {str(col): np.argmax(np.bincount(np.array(y_test)[cats == col]))}
    )


# Get the true labels associated with the majority prediction


# random_indices = random.sample(range(1000), images_to_plot)

random_indices = sample_idx
sample_images = X_test[random_indices]
sample_labels = cats[random_indices]
plt.clf()
plt.style.use("seaborn-muted")

fig, axes = plt.subplots(
    10,
    10,
    figsize=(15, 15),
    sharex=True,
    sharey=True,
    subplot_kw=dict(adjustable="box", aspect="equal"),
)  # https://stackoverflow.com/q/44703433/1870832

for i in range(images_to_plot):

    # axes (subplot) objects are stored in 2d array, accessed with axes[row,col]
    subplot_row = i // 10
    subplot_col = i % 10
    ax = axes[subplot_row, subplot_col]

    # plot image on subplot
    plottable_image = np.reshape(sample_images[i], (28, 28))
    ax.imshow(plottable_image, cmap="gray_r")

    ax.set_title("Pred Label: {}".format(cats_true[str(sample_labels[i][0])]))
    # ax.set_title('Predict Label: {}'.format(cats[i]))
    ax.set_xbound([0, 28])

plt.tight_layout()
plt.show()

#%%
# Plot the latent space

latent_vectors = chain_call(m1.latent_sample, X_test, 1000)


#%%
# verify sklearn gaussian mixture?
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.mixture import BayesianGaussianMixture

pca = PCA(2)
# pca = TSNE(2)
X_pca = pca.fit_transform(latent_vectors)
kmeans = BayesianGaussianMixture(10, tol=1e-6, max_iter=1000)
pred = kmeans.fit_predict(X_pca)
print(purity_score(y_test, pred))

#%%
df_latent = pd.DataFrame(
    {
        "x1": X_pca[:, 0],
        "x2": X_pca[:, 1],
        "cat": ["pred_{}".format(i) for i in y_test],
        "kmeans": ["pred_{}".format(i) for i in pred],
    }
)
plt.figure(figsize=(10, 10))
sns.scatterplot(data=df_latent, x="x1", y="x2", hue="cat")

plt.figure(figsize=(10, 10))
sns.scatterplot(data=df_latent, x="x1", y="x2", hue="kmeans")

#%%

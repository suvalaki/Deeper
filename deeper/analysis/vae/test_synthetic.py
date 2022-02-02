#%%
import itertools
import tensorflow as tf

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

from deeper.models.vae.model import VAE as model
from deeper.models.vae.train import train
from deeper.models.vae.utils import chain_call
from deeper.utils.metrics import purity_score

#%%
color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9.0, 5.0)
    plt.ylim(-3.0, 6.0)
    plt.xticks(())


#%% make data
state = np.random.RandomState(0)
X_reg = state.random((25, 10))
X_bool = (state.random((25, 3)) > 0.5).astype(float)
X_ord = (state.random((25, 5)) > 0.5).astype(float)
X_cat0 = np.zeros((25, 9))
for i, idx in enumerate(state.binomial(8, 0.5, 25)):
    X_cat0[i, idx] = 1
X_cat1 = np.zeros((25, 5))
for i, idx in enumerate(state.binomial(4, 0.5, 25)):
    X_cat1[i, idx] = 1
X_cat = np.concatenate([X_cat0, X_cat1], -1)
X = np.concatenate([X_reg, X_bool, X_ord, X_cat], -1)


#%% Fit a Gaussian mixture with EM using five components
gmm = mixture.GaussianMixture(n_components=5, covariance_type="full").fit(X)
y = gmm.predict(X)

# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

ohe = OneHotEncoder()
y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1))
y_test_ohe = ohe.fit_transform(y_test.reshape(-1, 1))


#%%
params = {
    "input_regression_dimension": X_train.shape[1],
    "input_boolean_dimension": 0,
    "input_ordinal_dimension": (X_ord.shape[1],),
    "input_categorical_dimension": 0,
    "output_regression_dimension": X_train.shape[1],
    "output_boolean_dimension": 0,
    "output_ordinal_dimension": (X_ord.shape[1],),
    "output_categorical_dimension": (y_train_ohe.shape[1],),
    "encoder_embedding_dimensions": [12, 12],
    "decoder_embedding_dimensions": [12, 12],
    "latent_dim": 64,
    "embedding_activations": tf.nn.relu,
    "bn_before": False,
    "bn_after": True,
    "latent_epsilon": 1e-12,
    "optimizer": tf.keras.optimizers.Adam(1e-3, epsilon=1e-16),
    "connected_weights": False,
    "latent_mu_embedding_dropout": 0.0,
    "latent_var_embedding_dropout": 0.0,
    "recon_dropouut": 0.0,
    #'latent_fixed_var': 10.0,
}

param_string = "vae" + "__".join([str(k) + "_" + str(v) for k, v in params.items()])

m1 = model(**params)

#%%
m1.compile()
m1.fit(X_train, X_train)


# %%
val = np.concatenate([X_train, y_train_ohe.todense()], 1)[0:10]
samp = m1.predict_one(X_train)


# %%
val = np.concatenate([X_train, y_train_ohe.todense()], 1)[0:10]
samp = m1.sample_one(X_train[0:10], val)

#%%
z_cooling = lambda: 1.0
train(
    m1,
    X_train,
    X_test,
    np.concatenate([X_train, y_train_ohe.todense()], 1),
    np.concatenate([X_test, y_test_ohe.todense()], 1),
    num=100,
    samples=1,
    epochs=1000,
    iter_train=1,
    num_inference=50,
    save="model_w_5",
    batch=True,
    save_results="./gumble_results.txt",
    beta_z_method=z_cooling,
    tensorboard=None,  # "./logs/" + param_string + "/samples__" + str(1),
)
# %% predict classes

pred = m1.predict_one(X_test)["x_recon_cat_groups_concat"].numpy().argmax(1)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
# %% Testing ordinal

# %%

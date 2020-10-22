#%%
import tensorflow as tf
import numpy as np

from deeper.models.vae.network import VaeNet


#%% Dummy Data
state = np.random.RandomState(0)
X_reg = state.random((25, 10))
X_bool = (state.random((25, 3)) > 0.5).astype(float)
X_ord = (state.random((25, 5)) > 0.5).astype(float)
X_cat0 = np.zeros((25, 9))
for i, idx in enumerate(state.binomial(9, 0.5, 25)):
    X_cat[i, idx] = 1
X_cat1 = np.zeros((25, 5))
for i, idx in enumerate(state.binomial(5, 0.5, 25)):
    X_cat[i, idx] = 1
X_cat = np.concatenate([X_cat0, X_cat1], -1)

X = np.concatenate([X_reg, X_bool, X_ord, X_cat], -1)

# %% Instantiate the model

net = VaeNet(
    X_reg.shape[1],
    X_bool.shape[1],
    (X_ord.shape[1],),
    (X_cat.shape[1],),
    X_reg.shape[1],
    X_bool.shape[1],
    (X_ord.shape[1],),
    (X_cat.shape[1],),
    (5, 5),
    (5, 5),
    10,
)

# %%

output = net(X)

# %%
VaeNet.output_names
# %%
net.output_names
# %%
net.call_dict(X)
# %%

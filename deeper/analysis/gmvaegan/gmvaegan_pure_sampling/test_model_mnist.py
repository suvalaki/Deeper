#%%

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from pathlib import Path
import tensorflow as tf
import numpy as np
from tqdm import tqdm

tf.random.set_seed(1234)
tf.keras.backend.set_floatx('float64')

import numpy as np
from deeper.models.gmvaegan.gmvaegan_pure_sampling import model
from deeper.models.gmvaegan.gmvaegan_pure_sampling.utils import (
    chain_call,
    chain_call_dataset,
    purity_score,
)
from deeper.models.gmvaegan.gmvaegan_pure_sampling.train import (
    train, 
    pretrain_with_clusters
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

y_ohe = OneHotEncoder()
y_train_ohe = np.array(y_ohe.fit_transform(y_train.reshape(-1,1)).todense())
y_test_ohe = np.array(y_ohe.transform(y_test.reshape(-1,1)).todense())


#%% Instantiate the model
from importlib import reload

model = reload(model)


initial_learning_rate = 1e-3
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10,
    decay_rate=0.8,
    staircase=True)


params = {
    "descriminator_dimensions":[512,512],
    "components":len(set(y_train)),
    "input_dimension":X_train.shape[1],
    "embedding_dimensions":[512, 512, ],
    "latent_dimensions":64,
    "mixture_embedding_dimensions":[512, 512,],
    "mixture_latent_dimensions":64,
    "embedding_activations":tf.nn.leaky_relu,
    "kind":"binary",
    "learning_rate":initial_learning_rate,
    "gradient_clip":1e2,
    "bn_before":False,
    "bn_after":False,
    "categorical_epsilon":0.0,
    "reconstruction_epsilon":0.0,
    "latent_epsilon":0.0,
    "latent_prior_epsilon":0.0,
    "z_kl_lambda":1.0,
    "c_kl_lambda":1.0,
    "cat_latent_bias_initializer":None,
    "optimizer":tf.keras.optimizers.Adam(initial_learning_rate, epsilon=1e-16),
    "categorical_latent_embedding_dropout":0.2,
    "mixture_latent_mu_embedding_dropout":0.2,
    "mixture_latent_var_embedding_dropout":0.2,
    "mixture_posterior_mu_dropout":0.2,
    "mixture_posterior_var_dropout":0.2,
    "recon_dropouut":0.2,
    #'latent_fixed_var': 10.0,
}

param_string = "__".join([str(k)+"_"+str(v) for k,v in params.items()])





m1 = model.GmvaeGan(**params)


#m1.load_weights('model_w')

#%% Examine SOftmax Distribution
import pandas as pd

logits, prob = m1.gmvae.graph_qy_g_x.call(X_test, training=False)

logit_df = pd.DataFrame()
for col in range(logits.shape[1]):
    temp_df = pd.DataFrame({'value':prob[:,col]})
    temp_df['k'] = str(col)
    logit_df = logit_df.append(temp_df, ignore_index=True)
import seaborn as sns
sns.violinplot(data=logit_df, y='value' , x='k')

# Plot the correct prediction densities



# confusion matrix of the classification
# for each caategory map the appropriate prediction
confusion_matrix(y_test, np.argmax(m1.predict(X_test),1))


#%% Pretrain the model with some known clusters
if False:
    pretrain_with_clusters(
        m1, 
        X_train, y_train, 
        X_test, y_test, 
        num=10, 
        samples=10,
        epochs=2, 
        iter_train=1, 
        num_inference=1000, 
        save='model_w',
    )


#%% setup cooling for trainign loop constants

#z_cooling = cooling.CyclicCoolingRegime(cooling.linear_cooling, 1e-1, 1, 25, 35)
#y_cooling = cooling.CyclicCoolingRegime(cooling.linear_cooling, 10.0, 1.0, 25, 35)

z_cooling = lambda: 1.0 
y_cooling = lambda: 1.0

#%% Train the model
# with tf.device('/gpu:0'):
train(
    m1, 
    X_train, y_train, 
    X_test, y_test, 
    num=100, 
    samples=1,
    epochs=1000, 
    iter_train=1, 
    num_inference=1000, 
    save='model_w',
    batch=True,
    temperature_function=lambda x: exponential_multiplicative_cooling(
        x, 0.5, 0.5, 0.98),
    save_results='./gumblevae_results.txt',
    beta_z_method=z_cooling,
    beta_y_method=y_cooling,
)


#%%
qy_g_x__logit, qy_g_x__prob = m1.graph_qy_g_x(X_train[[9]])
qy_g_x__ohe =np.array([
    m1.graph_qy_g_x_ohe(qy_g_x__prob, .005).numpy()[0]
    for i in range(1000)])

#%%
logit_df = pd.DataFrame()
for col in range(qy_g_x__ohe.shape[1]):
    temp_df = pd.DataFrame({'value':qy_g_x__ohe[:,col]})
    temp_df['k'] = str(col)
    logit_df = logit_df.append(temp_df, ignore_index=True)
import seaborn as sns
sns.boxplot(data=logit_df, y='value' , x='k')


#%%
for y in m1.entropy_fn(X_train[0:3]):
    print(tf.shape(y))

#%%

for y in m1.call(X_train[0:3]):
    print(tf.shape(y))

#%%
m1.call(X_train[0:3])

#%%

#latent_vectors = chain_call(m1.call, X_test, 100)[5]
latent_vectors = m1.call(X_test)[5]


#%%
# verify sklearn gaussian mixture?
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

#pca = PCA(2)
pca = TSNE(2)
X_pca = pca.fit_transform(latent_vectors)
kmeans = GaussianMixture(10, tol=1e-6, max_iter = 1000)
pred = kmeans.fit_predict(X_pca)
print(purity_score(y_test, pred))

#%%
df_latent = pd.DataFrame({
    'x1':X_pca[:,0], 
    'x2':X_pca[:,1], 
    'cat':['pred_{}'.format(i) for i in y_test],
    'kmeans':['pred_{}'.format(i) for i in pred]
})
plt.figure(figsize=(10,10))
sns.scatterplot(data=df_latent,x='x1',y='x2',hue='cat')

plt.figure(figsize=(10,10))
sns.scatterplot(data=df_latent,x='x1',y='x2',hue='kmeans')

#%%

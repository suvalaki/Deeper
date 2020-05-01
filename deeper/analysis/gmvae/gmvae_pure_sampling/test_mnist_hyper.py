import tensorflow as tf


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

from tensorboard.plugins.hparams import api as hp
from tensorboard.plugins import projector


import logging, os, json

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
import numpy as np
from tqdm import tqdm

from deeper.models.gmvae.gmvae_pure_sampling import model
from deeper.models.gmvae.gmvae_pure_sampling.utils import (
    chain_call,
    chain_call_dataset,
    purity_score,
)
from deeper.models.gmvae.gmvae_pure_sampling.train import (
    train,
)
from deeper.utils.cooling import exponential_multiplicative_cooling
import deeper.utils.cooling as cooling

from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import ParameterGrid
from itertools import product




#%% SET HYPER PARAMETERS
HP_seed = hp.HParam('seed', hp.Discrete([123,456,789]))
HP_components = hp.HParam('components', hp.Discrete([10,20,30]))
HP_encoder_dims = hp.HParam('encoder_dims', hp.Discrete([
    str(([512,512], 64)),
    str(([512, 512, 64], 16)),
    str(([512, 512, 64, 16], 2)),
    str(([512, 512,], 2))
]))
HP_mixture_dims = hp.HParam('mixture_dims', hp.Discrete([
    str(([512,512], 64)),
    str(([512, 512, 64], 16)),
    str(([512, 512, 256], 128))
]))
HP_bn = hp.HParam('BatchNorm', hp.Discrete(['none', 'before', 'after']))
HP_connected_weights = hp.HParam('connected_weights', hp.Discrete([False, True]))
HP_samples = hp.HParam('samples', hp.Discrete([1]))

METRIC_AMI_TRAIN = hp.Metric('ami_train', display_name='ami_train')
METRIC_AMI_TEST = hp.Metric('ami_test', display_name='ami_test')
METRIC_PURITY_TRAIN = hp.Metric('purity_train', display_name='purity_train')
METRIC_PURITY_TEST = hp.Metric('purity_test', display_name='purity_test')
METRIC_loss = hp.Metric('loss', display_name='loss')
METRIC_likelihood = hp.Metric('likelihood', display_name='likelihood')
METRIC_z_prior_entropy = hp.Metric('z_prior_entropy', display_name='z_prior_entropy')
METRIC_y_prior_entropy = hp.Metric('y_prior_entropy', display_name='z_prior_entropy')
METRIC_max_cluster_attachment_test= hp.Metric('max_cluster_attachment_test', display_name='max_cluster_attachment_test')

def para_grid(config_overrides):
    experiments = [
        dict(zip(config_overrides.keys(), value)) 
        for value in product(*config_overrides.values())
    ]
    return experiments

param_grid = para_grid({
    HP_seed: HP_seed.domain.values,
    HP_components: HP_components.domain.values, 
    HP_encoder_dims: HP_encoder_dims.domain.values, 
    HP_mixture_dims: HP_mixture_dims.domain.values, 
    HP_bn: HP_bn.domain.values, 
    HP_connected_weights: HP_connected_weights.domain.values, 
    HP_samples: HP_samples.domain.values, 
})


with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[
            HP_seed,
            HP_components,
            HP_encoder_dims,
            HP_mixture_dims,
            HP_bn,
            HP_connected_weights,
            HP_samples,
        ],
        metrics=[
            METRIC_AMI_TRAIN, METRIC_AMI_TEST,
            METRIC_PURITY_TRAIN, METRIC_PURITY_TEST,
            METRIC_max_cluster_attachment_test,
            METRIC_loss, METRIC_likelihood, METRIC_z_prior_entropy,
            METRIC_y_prior_entropy,
        ],
    )


def train_test_model(run_id, hparams, X_train, y_train, X_test, y_test):
    
    #hp.hparams(hparams) # record the values used in this trial
    seed = hparams[HP_seed]
    tf.random.set_seed(seed)
    params = {
        "components": hparams[HP_components],
        "input_dimension": X_train.shape[1],
        "embedding_dimensions": eval(hparams[HP_encoder_dims])[0],
        "latent_dimensions": eval(hparams[HP_encoder_dims])[1],
        "mixture_embedding_dimensions": eval(hparams[HP_mixture_dims])[0],
        "mixture_latent_dimensions": eval(hparams[HP_mixture_dims])[1],
        "embedding_activations": tf.nn.relu,
        "kind": "binary",
        "learning_rate": 1.0,
        "gradient_clip": None,
        "bn_before": True if hparams[HP_bn]=='before' else False,
        "bn_after": True if hparams[HP_bn]=='after' else False,
        "categorical_epsilon": 0.0,
        "reconstruction_epsilon": 0.0,
        "latent_epsilon": 0.0,
        "latent_prior_epsilon": 0.0,
        "z_kl_lambda": 1.0,
        "c_kl_lambda": 1.0,
        "cat_latent_bias_initializer": None,
        "connected_weights": hparams[HP_connected_weights],
        # "optimizer":tf.keras.optimizers.Adam(lr_schedule, epsilon=1e-16),
        "optimizer": tf.keras.optimizers.Adam(1e-3, epsilon=1e-16),
        "categorical_latent_embedding_dropout": 0.2,
        "mixture_latent_mu_embedding_dropout": 0.2,
        "mixture_latent_var_embedding_dropout": 0.2,
        "mixture_posterior_mu_dropout": 0.2,
        "mixture_posterior_var_dropout": 0.2,
        "recon_dropouut": 0.2,
        #'latent_fixed_var': 0.01,
    }

    z_cooling = lambda: 1.0
    y_cooling = lambda: 1.0

    m1 = model.Gmvae(**params)

    params["embedding_activations"] = "relu"
    params["optimizer"] = "adam_1e-3_1e-9"

    param_string = (
        "/seed__"
        + str(seed)
        + "/"
        + "/".join([str(k) + "_" + str(v) for k, v in params.items()])
    )

    train(
        m1,
        X_train,
        y_train,
        X_test,
        y_test,
        num=100,
        samples=hparams[HP_samples],
        epochs=110,
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
        tensorboard=run_id,
    )

    idx_tr = m1.predict(X_train).numpy().argmax(1)
    idx_te = m1.predict(X_test).numpy().argmax(1)
    
    ami_tr = adjusted_mutual_info_score(
        y_train, idx_tr, average_method="arithmetic"
    )
    ami_te = adjusted_mutual_info_score(
        y_test, idx_te, average_method="arithmetic"
    )

    attch_te = np.array(
        np.unique(idx_te, return_counts=True)[1]
    ).max() / len(idx_te)

    purity_train = purity_score(y_train, idx_tr)
    purity_test = purity_score(y_test, idx_te)

    return ami_tr, ami_te, purity_train, purity_test


#%% Load MNIST and make it binary encoded
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape(X_train.shape[0], 28 * 28)
X_test = X_test.reshape(X_test.shape[0], 28 * 28)
X_train = (X_train > 0.5).astype(float)
X_test = (X_test > 0.5).astype(float)


with open('gumple_result_search.csv' ,'w') as file:
    file.write(
        'run\tseed\tcomponents\tencoder_dims\tmixture_dims\tbn'
        '\tconnected_weights\tsamples'
        '\ttr_ami\tte_ami\ttr_pur\tte_pur'
    )


import pandas as pd
runs_done = pd.read_csv('gumple_result_search.csv', sep='\t')['run'].tolist()

#%% Gridsearch
session_num = 0
for i, hparams in enumerate(param_grid):
    if i not in runs_done:
        print(f'\n--- Starting Trial {i}')
        print({h: hparams[h] for h in hparams})

        run_name = f'run-{i}'

        ami_tr, ami_te, pur_tr, pur_te = train_test_model(
                'logs/hparam_tuning/' + run_name, 
                hparams, X_train, y_train, X_test, y_test
            )

        with tf.summary.create_file_writer('logs/hparam_tuning/' + run_name).as_default():

            hp.hparams(hparams)

            #tf.summary.scalar('ami_train', ami_tr, step=1)
            #tf.summary.scalar('te_ami', ami_te, step=1)
            #tf.summary.scalar('tr_pur', pur_tr, step=1)
            #tf.summary.scalar('te_pur', pur_te, step=1)

            with open('gumple_result_search.csv' ,'a') as file:
                file.write(
                    f'\n{i}'
                    f'\t{hparams[HP_seed]}'
                    f'\t{hparams[HP_components]}'
                    f'\t{hparams[HP_encoder_dims]}'
                    f'\t{hparams[HP_mixture_dims]}'
                    f'\t{hparams[HP_bn]}'
                    f'\t{hparams[HP_connected_weights]}'
                    f'\t{hparams[HP_samples]}'
                    f'\t{ami_tr}'
                    f'\t{ami_te}'
                    f'\t{pur_tr}'
                    f'\t{pur_te}'
                )

    session_num += 1
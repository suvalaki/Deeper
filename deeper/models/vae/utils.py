import tensorflow as tf
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
import io


def chain_call(func, x, num, scalar_dict={}):

    if type(x) != list and type(x) != tuple:
        x = [x]

    iters = x[0].shape[0] // num

    result = []
    j = 0
    while j * num < x[0].shape[0]:
        result = result + [
            func(
                *(y[j * num : min((j + 1) * num, y.shape[0])] for y in x),
                **scalar_dict
            )
        ]
        j += 1

    num_dats = len(result[0])
    # pivot the resultsif
    if type(result[0]) in [tuple, list]:
        result_pivot = [
            np.concatenate([y[i] for y in result], 0) for i in range(num_dats)
        ]
    elif type(result[0]) == dict:
        result_pivot = {
            k: np.concatenate([y[k] for y in result], 0)
            for k in result[0].keys()
        }
    else:
        result_pivot = np.concatenate(result, axis=0)
    return result_pivot


def plot_latent(latent_vectors):

    pca = PCA(2)
    X_pca = pca.fit_transform(latent_vectors)
    df_latent = pd.DataFrame(
        {
            "x1": X_pca[:, 0],
            "x2": X_pca[:, 1],
        }
    )

    f, (ax1) = plt.subplots(1, 1, sharey=True, figsize=(10, 10))
    ax1.scatter(
        df_latent.x1,
        df_latent.x2,
    )
    ax1.set_title("Latent Space")

    return f

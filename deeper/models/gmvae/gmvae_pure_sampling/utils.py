import tensorflow as tf
import numpy as np
from sklearn import metrics


def chain_call(func, x, num, scalar_dict={}):
    iters = x.shape[0] // num

    result = []
    j = 0
    while j * num < x.shape[0]:
        result = result + [
            func(x[j * num : min((j + 1) * num, x.shape[0])], **scalar_dict)
        ]
        j += 1

    num_dats = len(result[0])
    # pivot the resultsif
    if type(result[0]) in [tuple, list]:
        result_pivot = [
            np.concatenate([y[i] for y in result], 0) for i in range(num_dats)
        ]
    else:
        result_pivot = np.concatenate(result, axis=0)
    return result_pivot


@tf.function
def chain_call_dataset(func, dataset):

    result = []
    for x in dataset:
        result.append(func(x))

    if type(result[0]) == tuple:
        result_pivot = [
            np.concatenate([y[i] for y in result], 0) for i in range(num_dats)
        ]
    else:
        result_pivot = np.concatenate(result)
    return result_pivot


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(
        contingency_matrix
    )


def dataset_wrapper(X, SHUFFLE_BUFFER_SIZE=None, BATCH_SIZE=None):
    dataaset = tf.data.Dataset.from_tensor_slices(X)
    if SHUFFLE_BUFFER_SIZE is not None:
        dataaset.shuffle(SHUFFLE_BUFFER_SIZE)
    if BATCH_SIZE is not None:
        dataaset = dataaset.batch(BATCH_SIZE)


def numpy_tf_dataset(
    X_train, y_train, X_test, y_test, SHUFFLE_BUFFER_SIZE=None, BATCH_SIZE=None
):
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    return train_dataset, test_dataset

import numpy as np


def ohe_to_ordinalohe(data, axis=-1, threshold=0.5):
    dat = np.apply_along_axis(
        lambda x: [
            (x[:, :i] >= threshold).any() for i in range(2, x.shape[axis] + 1)
        ],
        1,
        data,
    )
    return dat

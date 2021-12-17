import numpy as np


def ohe_to_ordinalohe(data, axis=-1, threshold=0.5):
    dat = np.apply_along_axis(
        lambda x: [
            (x[:, i:] >= threshold).any() for i in range(1, x.shape[axis])
        ],
        1,
        data,
    )
    return dat


def index_to_ohe(index: int, max_index: int = 0) -> np.ndarray:
    if max_index == 0:
        return np.array([])
    x = np.zeros(max_index)
    x[index] = 1
    return x


class IndexOhe:
    def __init__(self, max_index: int):
        self.max_index = max_index

    def fit(self):
        # does nothing
        return self

    def transform(self, x):
        return index_to_ohe(x, self.max_index)
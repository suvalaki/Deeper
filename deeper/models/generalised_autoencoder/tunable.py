from typing import Dict
from deeper.optimizers.automl.tunable_types import (
    TunableType,
    TunableModelMixin,
    TunableActivation,
    TunableBoolean,
)
from itertools import chain
import scipy.interpolate
from collections.abc import Iterable


def reduce_sum(x):
    s = 0
    for z in x:
        if isinstance(z, Iterable):
            s += sum(z)
        else:
            s += z
    return s


def map_list_boundaries(hp, name, n_name, n_layers, lat_dim, max_dim):

    if n_layers is None:
        return []

    # Primary method creates a linear interpolation between max and min
    # dimensions such that hidden layers sizes are evenly space in between
    return interpolate_list(lat_dim + 1, max_dim, n_layers)


def get_embedding_linear_interpolation(in_dim, lat_dim, compression_layers):
    x = [0, compression_layers + 1]
    y = [round(lat_dim), round(in_dim)]
    y_interp = scipy.interpolate.interp1d(x, y)
    return [int(round(y_interp(i).tolist())) for i in range(1, compression_layers + 1)][::-1]


def get_embedding_fractional_interpolation(in_dim, lat_dim, compression_rate, compression_layers):
    x = [0, 1]
    y = [round(late_dim), round(in_dim)]
    y_interp = scipy.interpolate.interp1d(x, y)
    return [
        int(round(y_interp(compression_rate ** i).tolist()))
        for i in range(1, compression_layers + 1)
    ][::-1]


# TODO: Maybe use weakref


class TunableLatentDimensions(int, TunableType):

    _min: int = 1
    _max: int
    _default = 10
    _max_latent_fraction = 0.35
    _input_dimension: int  # Reference

    def update_backref(self, cn):
        self._input_dimension = reduce_sum(cn.input_dimensions.as_list())
        self._output_dimension = reduce_sum(cn.output_dimensions.as_list())
        self._max = round(self._max_latent_fraction * self._input_dimension)

    def tune_method(cls, hp, nm):
        return hp.Int(nm + "", cls._min, cls._max, default=cls._default)


class TunableEmbeddingDimensions(tuple, TunableType):

    _min = 0
    _max = 5
    _default = 0
    _input_dimension: int
    _output_dimension: int
    _latent_dim: TunableLatentDimensions  # Reference

    def update_backref(self, cn):
        self._input_dimension = reduce_sum(cn.input_dimensions.as_list())
        self._output_dimension = reduce_sum(cn.output_dimensions.as_list())
        self._latent_dim = cn.latent_dim

    def tune_method(self, hp, nm):
        layers = hp.Int(nm + "_layers", self._min, self._max, default=self._default)
        return get_embedding_linear_interpolation(self._input_dimension, self._latent_dim, layers)


class TunableDecodingDimensionsReflectReverse(TunableEmbeddingDimensions):

    _embedding_dimensions: TunableEmbeddingDimensions

    def update_backref(self, c):
        super().update_backref(c)
        if isinstance(c.encoder_embedding_dimensions, TunableEmbeddingDimensions):
            self._embedding_dimensions = c.encoder_embedding_dimensions
        else:
            raise "Expected TunableEmbeddingDimensions to mirror"

    def tune_method(self, hp, nm):
        encoder_prefix = nm[: -len("decoder_embedding_dimensions")] + "encoder_embedding_dimensions"
        # return self._embedding_dimensions
        return self._embedding_dimensions.tune_method(hp, encoder_prefix)[::-1]


def create_autoencoder_tunable_dims() -> Dict[str, TunableType]:

    ...
from __future__ import annotations
import tensorflow as tf

from typing import NamedTuple


class InputDisentangler:
    class DisentangledInputs(NamedTuple):
        inputs: NamedTuple
        temperature: NamedTuple
        weights: NamedTuple

    def __init__(self, weight_getter, **kwargs):
        self.__weight_getter = weight_getter

    def call_inputs(
        self, x, temp=None, weight=None, training=False
    ) -> InputDisentangler.DisentangledInputs:

        weights = self.__weight_getter(self.optimizer.iterations)
        if type(weights) == list:
            t, w = weights

            if not temp:
                temp = t
            if not weight:
                weight = w

        else:
            if not weight:
                weight = weights

        if temp is not None:
            inputs = (x, temp)
        else:
            inputs = x

        return InputDisentangler.DisentangledInputs(inputs, temp, weight)

    def call_test_inputs(self, x):
        ...


class ClusteringMixin(InputDisentangler):
    def __init__(self, weight_getter, network, cluster_parser, **kwargs):
        InputDisentangler.__init__(self, weight_getter)
        self.__cluster_parser = cluster_parser
        self.__network = network

    def call_cluster(self, x, temp=None, weight=None, training=False) -> tf.Tensor:
        ins = self.call_inputs(x, temp, weight, training)
        return self.__cluster_parser(self.__network(ins.inputs, training=training))

    @property
    def implements_clustering(self):
        return self.__cluster_parser is not None


class LatentMixin(InputDisentangler):
    def __init__(self, weight_getter, network, latent_parser, **kwargs):
        InputDisentangler.__init__(self, weight_getter)
        self.__latent_parser = latent_parser
        self.__network = network

    def call_latent(self, x, temp=None, weight=None, training=False) -> tf.Tensor:
        ins = self.call_inputs(x, temp, weight, training)
        return self.__latent_parser(self.__network(ins.inputs, training=training))

    @property
    def implements_latent(self):
        return self.__latent_parser is not None


class ReconstructionMixin(InputDisentangler):
    def __init__(self, weight_getter, network, reconstruction_parser, **kwargs):
        InputDisentangler.__init__(self, weight_getter)
        self.__reconstruction_parser = reconstruction_parser
        self.__network = network

    def call_reconstruction(self, x, temp=None, weight=None, training=False):
        ins = self.call_inputs(x, temp, weight, training)
        return self.__reconstruction_parser(self.__network(ins.inputs, training=training))

    @property
    def implements_reconstruction(self):
        return self.__reconstruction_parser is not None

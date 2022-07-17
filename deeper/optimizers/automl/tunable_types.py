import tensorflow as tf
from abc import ABC, abstractmethod
import typing
from pydantic import BaseModel


def pydantic_field_is_optional(field):
    return "Optional" in field._type_display()


def pydantic_tunable_optional(hp, field, nm):
    conditional_field = hp.Boolean()
    ...


class TunableType(ABC):
    @abstractmethod
    def tune_method(cls):
        ...

    def __repr__(self):
        return "Tunable(" + super().__repr__() + ")"


class TunableBoolean(int, TunableType):
    def tune_method(cls, hp, nm):
        conditional = hp.Boolean(nm + "_use")
        return conditional

    def __bool__(self):
        return self != 0

    def __len__(self):
        return False

    def __repr__(self):
        return "Tunable(" + str(bool(self)) + ")"


class TunableOptional(TunableType):

    _optional = True

    def tune_method(cls, hp, nm):

        conditional = hp.Boolean(nm + "_use", default=True)
        with hp.conditional_scope(nm + "_use", True):
            param = super().tune_method(hp, nm)

        return param


class TunableDropout(float, TunableType):

    _min = 0.05
    _max = 0.75
    _step = 0.05
    _default = 0.05
    _sampling = None

    def tune_method(cls, hp, nm):
        if cls._sampling is not None:
            return hp.Float(
                nm + "_dropout",
                cls._min,
                cls._max,
                default=cls._default,
                sampling=cls._sampling,
            )
        else:
            return hp.Float(
                nm + "_dropout",
                cls._min,
                cls._max,
                default=cls._default,
                step=cls._step,
            )


class OptionalTunableDropout(TunableOptional, TunableDropout):
    ...


class TunableRegularisationParam(TunableType):

    _min = 1e-6
    _max = 0.99
    _default = 0.1
    _sampling = "log"

    def __init__(self, name: str):
        self.nm = name

    def tune_method(cls, hp, nm):
        return hp.Float(
            nm + "_" + cls.nm,
            cls._min,
            cls._max,
            default=cls._default,
            sampling=cls._sampling,
        )


class OptionalTunableRegularisationParam(TunableOptional, TunableRegularisationParam):
    ...


class TunableL1L2Regulariser(tf.keras.regularizers.L1L2, TunableType):

    _l1 = OptionalTunableRegularisationParam("value")
    _l2 = OptionalTunableRegularisationParam("value")

    def tune_method(cls, hp, nm):
        l1 = cls._l1.tune_method(hp, nm + "_l1")
        l2 = cls._l2.tune_method(hp, nm + "_l2")
        return tf.keras.regularizers.L1L2(l1, l2)


class OptionalTunableL1L2Regulariser(TunableOptional, TunableL1L2Regulariser):
    ...


class TunableActivation(tf.keras.layers.Activation, TunableType):

    _activations = ["relu", "elu"]
    _default = "relu"

    def tune_method(cls, hp, nm):
        return tf.keras.layers.Activation(hp.Choice(nm, cls._activations, default=cls._default))


class TunableModelMixin(BaseModel):
    def parse_tunable(self, hp, prefix=""):
        kv = {}
        for nm, field in dict(self).items():
            if isinstance(field, TunableType):
                kv.update({nm: field.tune_method(hp, prefix + nm)})
            elif isinstance(field, TunableModelMixin):
                kv.update({nm: field.parse_tunable(hp, prefix + nm + "_")})
            else:
                kv.update({nm: field})
        return type(self)(**kv)


if __name__ == "__main__":

    import keras_tuner
    from deeper.layers.encoder import Encoder
    from deeper.optimizers.automl.tunable_types import TunableModelMixin

    class WrappedConfig(TunableModelMixin):
        hello: Encoder.Config

    hp = keras_tuner.HyperParameters()
    z = Encoder.Config()
    z2 = WrappedConfig(hello=z)

    zh = z2.parse_tunable(hp)

from __future__ import annotations
from typing import Tuple
import tensorflow as tf


class ExtensionTypeIterableMixin:

    _current = 0

    def __iter__(self):
        return (getattr(self, nm.name) for nm in self._tf_extension_type_fields())
        # for nm in self._tf_extension_type_fields():
        #     yield getattr(self, nm.name)

    def _asdict(self):
        return {nm.name: getattr(self, nm.name) for nm in self._tf_extension_type_fields()}

    def __getitem__(self, subscript):
        l = [x for x in self]
        return l.__getitem__(subscript)

    def __len__(self):
        return len(self._tf_extension_type_fields())

    @classmethod
    def extract_unpacked(cls, x, i, n, axis=0):
        if isinstance(x, tf.Tensor):
            return tf.unstack(x, n, axis=0)[i]

        return type(x)(
            **{
                k: cls.extract_unpacked(v, i, n, axis)
                if isinstance(v, tf.experimental.ExtensionType)
                else tuple(cls.extract_unpacked(z, i, n, axis) for z in v)
                if type(v) == tuple
                else cls.extract_unpacked(v, i, n, axis)
                for k, v in x._asdict().items()
            }
        )

    @classmethod 
    def reduce(
        cls, 
        x:Tuple[ExtensionTypeIterableMixin], 
        reducer=lambda x: tf.reduce_mean(tf.stack(x, axis=0), axis=0)
    ):

        if isinstance(x[0], tf.Tensor):
            return reducer(x)

        if isinstance(x[0], tuple):
            return tuple(
                cls.reduce([v[i] for v in x], reducer) 
                for i in range(len(x[0]))
            )
                                   
        return type(x[0])(
            **{
                k: cls.reduce([
                   getattr(v, k) for v in x], 
                   reducer
               )
               for k in x[0]._asdict().keys()
            }
        )


def factory_from_named_tuple(x):
    field_types = x.__annotations__

    class NewClass(tf.experimental.ExtensionType):
        __name__ = x.__name__ + "ExtensionType"
        for z, k in field_types.items():
            locals()[z] = k

    return Name

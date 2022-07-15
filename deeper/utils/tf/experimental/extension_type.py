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
        if type(x) == tf.Tensor:
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


def factory_from_named_tuple(x):
    field_types = x.__annotations__

    class NewClass(tf.experimental.ExtensionType):
        __name__ = x.__name__ + "ExtensionType"
        for z, k in field_types.items():
            locals()[z] = k

    return Name

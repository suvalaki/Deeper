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

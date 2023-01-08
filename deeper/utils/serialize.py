import json
import tensorflow as tf


# https://stackoverflow.com/questions/65338261/combine-multiple-json-encoders
class KerasObjectEncoder(json.JSONEncoder):
    """JSON serializer for objects not serializable by default json code"""

    def __call__(self, o):
        return self.default(o)

    def default(self, o):
        if isinstance(
            o,
            (tf.Module, tf.keras.layers.Layer, tf.keras.optimizers.Optimizer),
        ):
            return tf.keras.layers.serialize(o)
        return super().default(o)
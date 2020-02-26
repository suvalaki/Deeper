import tensorflow as tf
import json
from deeper.models.gmvae.gmvae_pure_sampling import model, train, utils
from deeper.utils.cooling import get_regime, CallableTempCls

optimizers = tf.keras.optimizers


def get_config():
    pass


def save_config():
    pass


class ModelWrapper(model.Gmvae):
    def get_config(self):
        func = model.gmvae.__init__
        params = [
            x
            for x in func.__code__.co_varnames[: func.__code__.co_argcount]
            if not x in ["self", "args", "kwargs"]
        ]
        model_args = {k: getattr(self, k) for k in params}
        activation_args = tf.keras.optimizer.get(self.activation)
        model_args["embedding_activations"] = activation_args
        optimizer_args = self.optimizer.get_config()
        model_args["optimizer"] = optimizer_args
        return model_args

    def build_from_config(self, config: dict):
        if "config" not in config["optimizer"]:
            config["optimizer"]["config"] = {}

        config["embedding_activations"] = tf.keras.activations.get(
            config["embedding_activations"]
        )

        config["optimizer"] = tf.keras.optimizers.get(config["optimizer"])
        return config

    @staticmethod
    def parse_train_config(config: dict):

        required_params = [
            "num",
            "samples",
            "epochs",
            "iter_train",
            "num_inference",
            "batch",
            "temperature_function",
            "beta_z_method",
            "beta_y_method",
            "save",
            "save_results",
            "tensorboard",
        ]

        # Parse Method functions
        method_params = [
            "temperature_function",
            "beta_z_method",
            "beta_y_method",
        ]
        for x in method_params:
            config[x] = get_regime(config[x])

        return config

    def train(self, *args, **kwargs):
        train.train(self, *args, **kwargs)

    def train_from_config(
        self, X_train, y_train, X_test, y_test, config: dict
    ):
        parsed_config = self.parse_train_config(config)
        self.train(X_train, y_train, X_test, y_test, **parsed_config)

    def __init__(self, config: dict):
        super(ModelWrapper, self).__init__(**self.build_from_config(config))

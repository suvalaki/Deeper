from __future__ import annotations
import tensorflow as tf

from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop

from deeper.models.gan.network import GanNet
from deeper.models.gan.network_loss import GanLossNet
from deeper.models.gan.descriminator import DescriminatorNet
from deeper.models.generalised_autoencoder.network import ModelConfigType

tfk = tf.keras
Model = tfk.Model


class Gan(Model):
    class Config(GanNet.Config):
        generator: ModelConfigType

        class Config:
            arbitrary_types_allowed = True
            smart_union = True

    Config.update_forward_refs()

    def __init__(self, config: Gan.Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.network = GanNet(config, **kwargs)
        self.lossnet = GanLossNet(**kwargs)
        self.weight_getter = self.config.generator.get_model_type().CoolingRegime(
            self.config.generator, dtype=self.dtype
        )

    def call(self, x, temp=None, training=False):

        if not temp:
            weights = self.weight_getter(self.optimizer.iterations)
            if type(weights) == list:
                temp, weight = weights

        inputs = (x, temp) if temp else x
        return self.network.fake_getter(self.network.generatornet(inputs, training=training))

    def train_step(self, data, training: bool = False):

        data = data_adapter.expand_1d(data)
        x, y = data

        temp = None
        weights = self.weight_getter(self.optimizer.iterations)
        if type(weights) == list:
            temp, weight = weights
        inputs = (x, temp) if temp else x

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            y_pred = self.network(inputs, y, training=True)
            gen_losses, descrim_losses = self.lossnet(y, y_pred, training=True)
            descrim_loss = tf.reduce_mean(descrim_losses)
            gen_loss = tf.reduce_mean(gen_losses)

        # Train the descriminator to identify real from fake samples
        self.optimizer.minimize(
            descrim_loss,
            self.network.descriminator.trainable_variables,
            tape=disc_tape,
        )

        # Train the generator to fool the descriminator
        self.optimizer.minimize(
            gen_loss,
            self.network.generatornet.trainable_variables,
            tape=gen_tape,
        )

        return {
            "loss/loss_generative": gen_loss,
            "loss/loss_descriminative": descrim_loss,
            **{"loss/" + v.name: v.result() for v in self.metrics if "accuracy" not in v.name},
            **{"acc/" + v.name: v.result() for v in self.metrics if "accuracy" in v.name},
        }

    def test_step(self, data):

        data = data_adapter.expand_1d(data)
        x, y = data

        temp = None
        weights = self.weight_getter(0)
        if type(weights) == list:
            temp, weight = weights

        inputs = (x, temp) if temp else x
        y_pred = self.network(inputs, y, training=False)
        gen_losses, descrim_losses = self.lossnet(y, y_pred, training=False)
        descrim_loss = tf.reduce_mean(descrim_losses)
        gen_loss = tf.reduce_mean(gen_losses)

        return {
            "loss/loss_generative": gen_loss,
            "loss/loss_descriminative": descrim_loss,
            **{"loss/" + v.name: v.result() for v in self.metrics if "accuracy" not in v.name},
            **{"acc/" + v.name: v.result() for v in self.metrics if "accuracy" in v.name},
        }

from __future__ import annotations
import tensorflow as tf

from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop
from functools import singledispatchmethod as overload

from deeper.models.gan.network import GanNet
from deeper.models.gan.network_loss import GanLossNet
from deeper.models.gan.descriminator import DescriminatorNet
from deeper.models.generalised_autoencoder.network import ModelConfigType
from deeper.utils.model_mixins import InputDisentangler, ReconstructionMixin, ClusteringMixin

tfk = tf.keras
Model = tfk.Model


class Gan(Model, ReconstructionMixin, ClusteringMixin):
    class Config(GanNet.Config):
        generator: ModelConfigType
        training_ratio: int = 3

        class Config:
            arbitrary_types_allowed = True
            smart_union = True

    Config.update_forward_refs()

    @overload
    def __init__(self, network: GanNet, config: Gan.Config, **kwargs):
        Model.__init__(self, **kwargs)
        self.network = network
        self.config = config
        self.lossnet = GanLossNet(**kwargs)
        self.weight_getter = self.config.generator.get_model_type().CoolingRegime(
            self.config.generator, dtype=self.dtype
        )
        ReconstructionMixin.__init__(
            self,
            self.weight_getter,
            self.network.generatornet,
            self.network.fake_getter,
        )
        ClusteringMixin.__init__(
            self,
            self.weight_getter,
            self.network.generatornet,
            config.generator.get_cluster_output_parser_type()()
            if hasattr(config.generator, "get_cluster_output_parser_type")
            else None,
        )

    @__init__.register
    def from_config(self, config: BaseModel, **kwargs):
        network = GanNet(config, **kwargs)
        self.__init__(network, config, **kwargs)

    def call(self, x, temp=None, training=False):
        return self.call_reconstruction(x, temp, training)

    def train_step(self, data, training: bool = False):

        data = data_adapter.expand_1d(data)
        x, y = data
        inputs, temp, weights = self.call_inputs(x)

        # Use a single pass over the network for efficiency.
        # Normaly would sequentially call generative and then descrimnative nets
        # Take multiple passes of the descriminator player according to 4.4 of
        # https://arxiv.org/pdf/1701.00160.pdf to ballance G and D.
        for i in range(self.config.training_ratio):
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
        if temp is not None:
            inputs = (x, temp)
        else:
            inputs = x

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

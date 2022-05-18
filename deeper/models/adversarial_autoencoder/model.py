from __future__ import annotations
import tensorflow as tf

from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop
from functools import singledispatchmethod as overload
from pydantic import BaseModel

from deeper.models.adversarial_autoencoder.network import (
    AdversarialAutoencoderNet,
)
from deeper.models.adversarial_autoencoder.network_loss import (
    AdverasrialAutoencoderLossNet,
)
from deeper.models.gan.descriminator import DescriminatorNet
from deeper.models.generalised_autoencoder.network import (
    ModelConfigType,
)
from deeper.models.generalised_autoencoder.base import (
    AutoencoderModelBaseMixin,
)
from deeper.utils.model_mixins import ClusteringMixin

tfk = tf.keras
Model = tfk.Model


class AdversarialAutoencoder(Model, AutoencoderModelBaseMixin, ClusteringMixin):
    class Config(AdversarialAutoencoderNet.Config):
        generator: ModelConfigType
        training_ratio: int = 1

        class Config:
            arbitrary_types_allowed = True
            smart_union = True

    Config.update_forward_refs()

    @overload
    def __init__(
        self, network: AdversarialAutoencoderNet, config: AdversarialAutoencoder.Config, **kwargs
    ):
        Model.__init__(self, **kwargs)
        self.config = config
        self.network = network
        self.lossnet = AdverasrialAutoencoderLossNet(config, self.network.generatornet, **kwargs)
        self.weight_getter = self.config.generator.get_model_type().CoolingRegime(
            self.config.generator, dtype=self.dtype
        )
        self.output_parser = config.generator.get_fake_output_getter()()
        self.latent_parser = config.generator.get_adversarialae_fake_output_getter()()
        AutoencoderModelBaseMixin.__init__(
            self,
            self.weight_getter,
            self.network.generatornet,
            config.generator.get_adversarialae_fake_output_getter()(),
            config.generator.get_fake_output_getter()(),
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
        network = AdversarialAutoencoderNet(config, **kwargs)
        self.__init__(network, config, **kwargs)

    def call(self, x, temp=None, training=False):

        inputs, temp, weights = self.call_inputs(x)
        return self.output_parser(self.network.generatornet(inputs, training=training))

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
                gen_losses, descrim_losses, recon_losses = self.lossnet(
                    self.lossnet.Input.from_output(
                        self.network,
                        self.lossnet.generator_lossnet,
                        y,
                        y_pred,
                        weights,
                    ),
                    training=True,
                )
                descrim_loss = tf.reduce_mean(descrim_losses)
                gen_loss = tf.reduce_mean(gen_losses)
                recon_loss = tf.reduce_mean(recon_losses)
                loss = tf.reduce_mean(gen_loss) + tf.reduce_mean(recon_loss)

            # Train the descriminator to identify real from fake samples
            self.optimizer.minimize(
                descrim_loss,
                self.network.descriminator.trainable_variables,
                tape=disc_tape,
            )

        # Train the generator to fool the descriminator
        # self.optimizer.minimize(
        #    loss,
        #   self.network.generatornet.trainable_variables,
        #    tape=gen_tape,
        # )

        # instead of doing this maybe train only encoder on GAN generator and
        # train ae decoder on this step.
        with tf.GradientTape() as recon_tape:
            y_pred = self.network(inputs, y, training=True)
            gen_losses, descrim_losses, recon_losses = self.lossnet(
                self.lossnet.Input.from_output(
                    self.network,
                    self.lossnet.generator_lossnet,
                    y,
                    y_pred,
                    weights,
                ),
                training=True,
            )
            descrim_loss = tf.reduce_mean(descrim_losses)
            gen_loss = tf.reduce_mean(gen_losses)
            recon_loss = tf.reduce_mean(recon_losses)
            loss = tf.reduce_mean(gen_loss) + tf.reduce_mean(recon_loss)

        # Train the reconstruction network
        self.optimizer.minimize(
            loss,
            self.network.generatornet.trainable_variables,
            tape=recon_tape,
        )

        return {
            "loss/loss_generative": gen_loss,
            "loss/loss_descriminative": descrim_loss,
            "loss/loss_reconstruction": recon_loss,
            **{"loss/" + v.name: v.result() for v in self.metrics if "accuracy" not in v.name},
            **{"acc/" + v.name: v.result() for v in self.metrics if "accuracy" in v.name},
        }

    def test_step(self, data):

        data = data_adapter.expand_1d(data)
        x, y = data

        inputs, temp, weights = self.call_inputs(x)
        if temp is not None:
            inpputs = (x, 0.5)

        y_pred = self.network(inputs, y, training=False)
        gen_losses, descrim_losses, recon_losses = self.lossnet(
            self.lossnet.Input.from_output(
                self.network,
                self.lossnet.generator_lossnet,
                y,
                y_pred,
                weights,
            ),
            training=True,
        )
        descrim_loss = tf.reduce_mean(descrim_losses)
        gen_loss = tf.reduce_mean(gen_losses)
        recon_loss = tf.reduce_mean(recon_losses)

        return {
            "loss/loss_generative": gen_loss,
            "loss/loss_descriminative": descrim_loss,
            "loss/loss_reconstruction": recon_loss,
            **{"loss/" + v.name: v.result() for v in self.metrics if "accuracy" not in v.name},
            **{"acc/" + v.name: v.result() for v in self.metrics if "accuracy" in v.name},
        }

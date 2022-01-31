from __future__ import annotations
import tensorflow as tf

from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop

from deeper.models.adversarial_autoencoder.network import (
    AdversarialAuoencoderNet,
)
from deeper.models.adversarial_autoencoder.network_loss import (
    AdverasrialAutoencoderLossNet,
)
from deeper.models.gan.descriminator import DescriminatorNet
from deeper.models.generalised_autoencoder.network import ModelConfigType

tfk = tf.keras
Model = tfk.Model


class AdversarialAutoencoder(Model):
    class Config(AdversarialAuoencoderNet.Config):
        generator: ModelConfigType
        training_ratio: int = 3

        class Config:
            arbitrary_types_allowed = True
            smart_union = True

    Config.update_forward_refs()

    def __init__(self, config: AdversarialAutoencoder.Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.network = AdversarialAuoencoderNet(config, **kwargs)
        self.lossnet = AdverasrialAutoencoderLossNet(config, self.network.generatornet, **kwargs)
        self.weight_getter = self.config.generator.get_model_type().CoolingRegime(
            self.config.generator, dtype=self.dtype
        )
        self.output_parser = config.generator.get_fake_output_getter()()

    def call(self, x, temp=None, training=False):

        if not temp:
            weights = self.weight_getter(self.optimizer.iterations)
            if type(weights) == list:
                temp, weight = weights
        if temp is not None:
            inputs = (x, temp)
        else:
            intputs = x

        return self.output_parser(self.network.generatornet(inputs, training=training))

    def train_step(self, data, training: bool = False):

        data = data_adapter.expand_1d(data)
        x, y = data

        temp = None
        weights = self.weight_getter(self.optimizer.iterations)
        if type(weights) == list:
            temp, weights = weights
        if temp is not None:
            inputs = (x, temp)
        else:
            intputs = x

        # Use a single pass over the network for efficiency.
        # Normaly would sequentially call generative and then descrimnative nets
        # Take multiple passes of the descriminator player according to 4.4 of
        # https://arxiv.org/pdf/1701.00160.pdf to ballance G and D.
        for i in range(self.config.training_ratio):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as recon_tape:

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

        # Train the reconstruction network
        self.optimizer.minimize(
            recon_loss,
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

        temp = None
        weights = self.weight_getter(0)
        if type(weights) == list:
            temp, weights = weights
        if temp is not None:
            inputs = (x, temp)
        else:
            intputs = x

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

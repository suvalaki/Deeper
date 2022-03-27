from __future__ import annotations

import tensorflow as tf
from typing import NamedTuple, Callable, Union
from pydantic import BaseModel, Field

from deeper.models.gan.descriminator import DescriminatorNet
from deeper.models.gan.base_getter import (
    BaseGanFakeOutputGetter,
    BaseGanRealOutputGetter,
)
from deeper.models.generalised_autoencoder.network import ConfigType
from deeper.models.gan.base_getter import GanTypeGetter


class GanGenerativeNet(tf.keras.layers.Layer):
    class Output(NamedTuple):
        generated: NamedTuple
        fake_descriminant: tf.Tensor

    def __init__(
        self,
        descriminatornet: DescriminatorNet,
        generatornet: tf.keras.layers.Layer,
        fake_output_getter: BaseGanFakeOutputGetter,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.descriminator = descriminatornet
        self.generatornet = generatornet
        self.fake_getter = fake_output_getter

    def call_post_generation(self, x, y_pred, training=False):
        y_pred_out = self.fake_getter(y_pred, training=training)
        fake_descrim = self.descriminator(y_pred_out, training=training)
        return GanGenerativeNet.Output(y_pred, fake_descrim)

    def call(self, x, training=False):
        y_pred = self.generatornet(x, training=training)
        return self.call_post_generation(x, y_pred, training=training)


class GanDescriminativeNet(tf.keras.layers.Layer):
    class Output(NamedTuple):
        generated: NamedTuple
        fake_descriminant: tf.Tensor
        real_descriminant: tf.Tensor

    def __init__(
        self,
        descriminatornet: DescriminatorNet,
        generatornet: tf.keras.layers.Layer,
        fake_output_getter: BaseGanFakeOutputGetter,
        real_output_getter: BaseGanRealOutputGetter,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.descriminator = descriminatornet
        self.generatornet = generatornet
        self.fake_getter = fake_output_getter
        self.real_getter = real_output_getter

    def call_post_generation(self, x, y, y_pred, training=False):
        y_pred_out = self.fake_getter(y_pred, training=training)
        fake_descrim = self.descriminator(y_pred_out, training=training)
        real_descrim = self.descriminator(
            self.real_getter(x, y, y_pred, training=training),
            training=training,
        )
        return GanDescriminativeNet.Output(y_pred, fake_descrim, real_descrim)

    def call(self, x, y, training=False):
        y_pred = self.generatornet(x, training=training)
        return self.call_post_generation(x, y, y_pred, training=training)


class GanNet(tf.keras.layers.Layer):
    class Config(BaseModel):
        descriminator: DescriminatorNet.Config
        generator: ConfigType

        class Config:
            allow_arbitrary_types = True
            smart_union = True

    Config.update_forward_refs()

    # Goal will be to max logprob while fooling the descimintaor
    class Output(NamedTuple):
        descriminative: GanDescriminativeNet.Output
        generative: GanGenerativeNet.Output

    def __init__(
        self, config: GanNet.Config, fake_getter=None, real_getter=None, generatornet=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.descriminator = DescriminatorNet(config.descriminator)
        self.generatornet = (
            generatornet
            if generatornet
            else config.generator.get_generatornet_type()(config.generator)
        )
        self.fake_getter = (
            fake_getter if fake_getter else config.generator.get_fake_output_getter()()
        )
        self.real_getter = (
            real_getter if real_getter else config.generator.get_real_output_getter()()
        )

        self.gan_generative = GanGenerativeNet(
            self.descriminator, self.generatornet, self.fake_getter
        )
        self.gan_descriminative = GanDescriminativeNet(
            self.descriminator,
            self.generatornet,
            self.fake_getter,
            self.real_getter,
        )

    def call_generative(self, x, training=False):
        return self.gan_generative(x, training=training)

    def call_descriminative(self, x, y, training=False):
        return self.gan_descriminative(x, y, training=training)

    def call(self, x, y, training=False):
        y_pred = self.generatornet(x, training=training)
        generative = self.gan_generative.call_post_generation(x, y_pred, training=training)
        descriminative = self.gan_descriminative.call_post_generation(
            x, y, y_pred, training=training
        )
        return GanNet.Output(descriminative, generative)

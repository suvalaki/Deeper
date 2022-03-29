from __future__ import annotations

import tensorflow as tf
from typing import NamedTuple
from pydantic import BaseModel, Field

from deeper.models.generalised_autoencoder.base import AutoencoderBase
from deeper.models.gan.network import GanNet


class AdversarialAuoencoderNet(GanNet, AutoencoderBase):
    class Config(GanNet.Config):
        ...

    Config.update_forward_refs()

    class Output(NamedTuple):
        descriminative: GanDescriminativeNet.Output
        generative: GanGenerativeNet.Output
        reconstruction: NamedTuple

    def __init__(self, config: AdversarialAuoencoderNet.Config, **kwargs):
        super().__init__(
            config,
            config.generator.get_adversarialae_fake_output_getter()(),
            config.generator.get_adversarialae_real_output_getter()(),
            **kwargs
        )

    def split_outputs(self, y) -> SplitCovariates:
        return self.generatornet.split_outputs(y)

    def call_ae_generator(self, x, training=False):
        return self.generatornet(x, training=training)

    def call(self, x, y, training=False):
        y_pred = self.call_ae_generator(x, training=False)
        generative = self.call_generative_post_generation(x, y_pred, training=training)
        descriminative = self.call_descriminative_post_generation(x, y, y_pred, training=training)
        return AdversarialAuoencoderNet.Output(descriminative, generative, y_pred)

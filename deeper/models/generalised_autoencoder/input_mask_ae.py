from __future__ import annotations

import tensorflow as tf

from pydantic import BaseModel, Field

from deeper.models.generalised_autoencoder.base import (
    AutoencoderTypeGetterBase,
    AutoencoderBase,
    AutoencoderModelBaseMixin,
)

# mask an input only layer and make it look like an autoencoder input,
# Include a latent parser to return the input value


# basically just need to combine the inputs into a single layer
class InputMaskAeNet(AutoencoderBase):
    class Config(BaseModel):

        input_dimensions: MultipleObjectiveDimensions = Field()
        output_dimensions: MultipleObjectiveDimensions = Field()

        class Config:
            arbitrary_types_allowed = True
            smart_union = True

    ...
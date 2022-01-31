from __future__ import annotations

import tensorflow as tf

from deeper.layers.binary import SigmoidEncoder
from pydantic import BaseModel, Field


class DescriminatorNet(SigmoidEncoder):
    def __init__(self, config: Descriminator.Config, **kwargs):
        super().__init__(config, **kwargs)

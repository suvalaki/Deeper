from __future__ import annotations
import tensorflow as tf
from typing import Sequence

from pydantic import BaseModel, Field


class MultipleObjectiveDimensions(BaseModel):
    regression: int = Field()
    boolean: int = Field()
    ordinal: Sequence[int] = Field()
    categorical: Sequence[int] = Field()

    def as_list(self):
        return [self.regression, self.boolean, self.ordinal, self.categorical]

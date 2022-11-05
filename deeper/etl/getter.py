from abc import ABC, abstractmethod
from datetime import datetime
from typing import Generic, TypeVar, NamedTuple, Tuple, Union, Sequence
import pandas as pd


## \defgroup getter_base Base Cache Getters
# \ingroup cache
#
# \subsection cache_getter_base Base Getter Interface
# Data loaders implement a get method in the format of one of the abstract
# getter interfaces.
#
# \{


OutputData = TypeVar("OutputData", bound=NamedTuple)


class DataExtractor(ABC, Generic[OutputData]):
    @abstractmethod
    def get(self, *args, **kwargs) -> OutputData:
        pass

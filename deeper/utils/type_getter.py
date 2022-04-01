from abc import ABC, abstractmethod


class NetworkTypeGetterBase(ABC):
    @classmethod
    @abstractmethod
    def get_network_type(self):
        pass

    @classmethod
    @abstractmethod
    def get_lossnet_type(self):
        pass

    @classmethod
    @abstractmethod
    def get_model_type(self):
        pass

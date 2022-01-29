from abc import ABC, abstractmethod


class NetworkTypeGetterBase(ABC):
    @abstractmethod
    def get_network_type(self):
        pass

    @abstractmethod
    def get_lossnet_type(self):
        pass

    @abstractmethod
    def get_model_type(self):
        pass

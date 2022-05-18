class GanTypeGetterMixin:
    @classmethod
    def get_network_type(self):
        from deeper.models.gan.network import (
            GanNet,
        )

        return GanNet

    @classmethod
    def get_lossnet_type(self):
        from deeper.models.gan.network_loss import (
            GanLossNet,
        )

        return GanLossNet

    @classmethod
    def get_model_type(self):
        from deeper.models.gan.model import (
            Gan,
        )

        return Gan

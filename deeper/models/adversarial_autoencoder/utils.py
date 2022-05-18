class AdversarialAutoencoderTypeGetterMixin:
    @classmethod
    def get_network_type(self):
        from deeper.models.adversarial_autoencoder.network import (
            AdversarialAutoencoderNet,
        )

        return AdversarialAutoencoderNet

    @classmethod
    def get_lossnet_type(self):
        from deeper.models.adversarial_autoencoder.network_loss import (
            AdverasrialAutoencoderLossNet,
        )

        return AdverasrialAutoencoderLossNet

    @classmethod
    def get_model_type(self):
        from deeper.models.adversarial_autoencoder.model import (
            AdversarialAutoencoder,
        )

        return AdversarialAutoencoder

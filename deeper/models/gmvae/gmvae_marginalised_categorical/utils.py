from deeper.models.generalised_autoencoder.base import (
    AutoencoderTypeGetterBase,
)


class StackedGmvaeTypeGetter(AutoencoderTypeGetterBase):
    def get_network_type(self):
        from deeper.models.gmvae.gmvae_marginalised_categorical.network import (
            StackedGmvaeNet,
        )

        return StackedGmvaeNet

    def get_lossnet_type(self):
        from deeper.models.gmvae.gmvae_marginalised_categorical.network_loss import (
            StackedGmvaeLossNet,
        )

        return StackedGmvaeLossNet

    def get_model_type(self):
        from deeper.models.gmvae.gmvae_marginalised_categorical.model import (
            StackedGmvae,
        )

        return StackedGmvae

    def get_latent_parser_type(self):
        from deeper.models.gmvae.gmvae_marginalised_categorical.latent import (
            StackedGmvaeLatentParser,
        )

        return StackedGmvaeLatentParser

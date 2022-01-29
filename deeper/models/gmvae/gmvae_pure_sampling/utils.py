from deeper.models.generalised_autoencoder.base import (
    AutoencoderTypeGetterBase,
)


class GumbleGmvaeTypeGetter(AutoencoderTypeGetterBase):
    def get_network_type(self):
        from deeper.models.gmvae.gmvae_pure_sampling.network import (
            GumbleGmvaeNet,
        )

        return GumbleGmvaeNet

    def get_lossnet_type(self):
        from deeper.models.gmvae.gmvae_pure_sampling.network_loss import (
            GumbleGmvaeNetLossNet,
        )

        return GumbleGmvaeNetLossNet

    def get_model_type(self):
        from deeper.models.gmvae.gmvae_pure_sampling.model import GumbleGmvae

        return GumbleGmvae

    def get_latent_parser_type(self):
        from deeper.models.gmvae.gmvae_pure_sampling.latent import (
            GumbleGmvaeLatentParser,
        )

        return GumbleGmvaeLatentParser

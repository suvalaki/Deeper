from deeper.models.generalised_autoencoder.base import (
    AutoencoderTypeGetterBase,
)
from deeper.models.gan.base_getter import GanTypeGetter
from deeper.models.adversarial_autoencoder.base_getter import (
    AdversarialAutoencoderTypeGetter,
)


class GumbleGmvaeTypeGetter(
    AutoencoderTypeGetterBase, GanTypeGetter, AdversarialAutoencoderTypeGetter
):
    @classmethod
    def get_cooling_regime(self):
        from deeper.models.gmvae.gmvae_pure_sampling import GumbleGmvae

        return GumbleGmvae.CoolingRegime

    @classmethod
    def get_network_type(self):
        from deeper.models.gmvae.gmvae_pure_sampling.network import (
            GumbleGmvaeNet,
        )

        return GumbleGmvaeNet

    @classmethod
    def get_lossnet_type(self):
        from deeper.models.gmvae.gmvae_pure_sampling.network_loss import (
            GumbleGmvaeNetLossNet,
        )

        return GumbleGmvaeNetLossNet

    @classmethod
    def get_model_type(self):
        from deeper.models.gmvae.gmvae_pure_sampling.model import GumbleGmvae

        return GumbleGmvae

    @classmethod
    def get_latent_parser_type(self):
        from deeper.models.gmvae.gmvae_pure_sampling.latent import (
            GumbleGmvaeLatentParser,
        )

        return GumbleGmvaeLatentParser

    @classmethod
    def get_cluster_output_parser_type(self):
        from deeper.models.gmvae.gmvae_pure_sampling.parser import (
            ClusterPredictionParser,
        )

        return ClusterPredictionParser

    # Gan getters Mixin

    @classmethod
    def get_generatornet_type(self):
        from deeper.models.gmvae.gmvae_pure_sampling.network import (
            GumbleGmvaeNet,
        )

        return GumbleGmvaeNet

    @classmethod
    def get_real_output_getter(self):
        from deeper.models.gmvae.gmvae_pure_sampling.parser import InputParser

        return InputParser

    @classmethod
    def get_fake_output_getter(self):
        from deeper.models.gmvae.gmvae_pure_sampling.parser import OutputParser

        return OutputParser

    # Adversarial Autoencoder getters Mixin

    @classmethod
    def get_adversarialae_real_output_getter(self):
        from deeper.models.gmvae.gmvae_pure_sampling.parser import (
            LatentPriorParser,
        )

        return LatentPriorParser

    @classmethod
    def get_adversarialae_fake_output_getter(self):
        from deeper.models.gmvae.gmvae_pure_sampling.parser import (
            LatentPosteriorParser,
        )

        return LatentPosteriorParser

    @classmethod
    def get_adversarialae_recon_loss_getter(self):
        from deeper.models.gmvae.gmvae_pure_sampling.parser import (
            ReconstructionOnlyLossOutputParser,
        )

        return ReconstructionOnlyLossOutputParser

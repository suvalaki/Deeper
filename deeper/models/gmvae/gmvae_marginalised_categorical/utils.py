from deeper.models.generalised_autoencoder.base import (
    AutoencoderTypeGetterBase,
)
from deeper.models.gan.base_getter import GanTypeGetter
from deeper.models.adversarial_autoencoder.base_getter import AdversarialAutoencoderTypeGetter


class StackedGmvaeTypeGetter(
    AutoencoderTypeGetterBase, GanTypeGetter, AdversarialAutoencoderTypeGetter
):
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

    # Gan getters Mixin

    def get_generatornet_type(self):
        from deeper.models.gmvae.gmvae_marginalised_categorical.network import (
            StackedGmvaeNet,
        )

        return StackedGmvaeNet

    def get_real_output_getter(self):
        from deeper.models.gmvae.gmvae_marginalised_categorical.parser import (
            InputParser,
        )

        return InputParser

    def get_fake_output_getter(self):
        from deeper.models.gmvae.gmvae_marginalised_categorical.parser import (
            OutputParser,
        )

        return OutputParser

    # Adversarial Autoencoder getters Mixin

    def get_adversarialae_real_output_getter(self):
        from deeper.models.gmvae.gmvae_marginalised_categorical.parser import (
            LatentPriorParser,
        )

        return LatentPriorParser

    def get_adversarialae_fake_output_getter(self):
        from deeper.models.gmvae.gmvae_marginalised_categorical.parser import (
            LatentPosteriorParser,
        )

        return LatentPosteriorParser

    def get_adversarialae_recon_loss_getter(self):
        from deeper.models.gmvae.gmvae_marginalised_categorical.parser import (
            ReconstructionOnlyLossOutputParser,
        )

        return ReconstructionOnlyLossOutputParser

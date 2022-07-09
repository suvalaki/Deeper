from typing import Tuple, Union

from deeper.layers.data_splitter import SplitCovariates, split_inputs, unpack_dimensions
from deeper.models.autoencoder import Autoencoder, AutoencoderLossNet, AutoencoderNet
from deeper.models.generalised_autoencoder.network import (
    GeneralisedAutoencoderNet,
    GumbleGmvaeNet,
    StackedGmvaeNet,
    VaeNet,
)
from deeper.models.gmvae.gmvae_marginalised_categorical import (
    StackedGmvaeLatentParser,
    StackedGmvaeLossNet,
    StackedGmvaeNet,
)
from deeper.models.gmvae.gmvae_pure_sampling import (
    GumbleGmvaeLatentParser,
    GumbleGmvaeNet,
    GumbleGmvaeNetLossNet,
)
from deeper.models.identity import IdentityLossNet, IdentityNet
from deeper.models.null import Null, NullLossNet, NullNet
from deeper.models.vae import VaeLatentParser, VaeLossNet, VaeNet

InputWeightTypes = Union[
    NullLossNet.InputWeight,
    IdentityLossNet.InputWeight,
    AutoencoderLossNet.InputWeight,
    VaeLossNet.InputWeight,
    StackedGmvaeLossNet.InputWeight,
    GumbleGmvaeNetLossNet.InputWeight,
]

AutoencoderConfigType = Union[
    NullNet.Config,
    IdentityNet.Config,
    AutoencoderNet.Config,
    StackedGmvaeNet.Config,
    GumbleGmvaeNet.Config,
    VaeNet.Config,
]

AutoencoderOutputType = Union[
    NullNet.Output,
    IdentityNet.Output,
    AutoencoderNet.Output,
    StackedGmvaeNet.Output,
    GumbleGmvaeNet.Output,
    VaeNet.Output,
]

AutoencoderLossInputType = Union[
    NullLossNet.Input,
    IdentityLossNet.Input,
    AutoencoderLossNet.Input,
    VaeLossNet.Input,
    StackedGmvaeLossNet.Input,
    GumbleGmvaeNetLossNet.Input,
]

AutoencoderLossOutputType = Union[
    NullLossNet.Output,
    IdentityLossNet.Output,
    AutoencoderLossNet.Output,
    VaeLossNet.Output,
    StackedGmvaeLossNet.Output,
    GumbleGmvaeNetLossNet.Output,
]

TupAutoencoderLossOutputType = Union[
    Tuple[NullLossNet.Output, ...],
    Tuple[IdentityLossNet.Output, ...],
    Tuple[AutoencoderLossNet.Output, ...],
    Tuple[VaeLossNet.Output, ...],
    Tuple[StackedGmvaeLossNet.Output, ...],
    Tuple[GumbleGmvaeNetLossNet.Output, ...],
]

TempAndWeightTypes = Tuple[tf.Tensor, InputWeightTypes]

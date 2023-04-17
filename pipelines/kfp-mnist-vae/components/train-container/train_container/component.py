# Container components require all of the code to already be
# present as functionality from the base_image. It is not possible
# to add additional code to the container image.

from kfp import dsl
from kfp.dsl import Input, Dataset

# This works with v2 API only


@dsl.container_component
def train_mnist_vae_model(
    data_locn: Input[Dataset],
    latent_dim: int,
    batch_size: int,
):
    """Create and train a VAE over the given MNIST data location"""

    return dsl.ContainerSpec(
        image="suvalaki/deeper:latest",
        command=[
            "sh",
            "-c"
            """python ./deeper/analysis/vae/test_model_mnist_cli.py \
                    --data_locn=$0 \
                    --latent_dim=$1 \
                    --batch_size=$2
            """,
        ],
        args=[data_locn.path, latent_dim, batch_size],
    )

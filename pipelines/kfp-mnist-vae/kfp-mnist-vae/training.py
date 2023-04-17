import kfp
from kfp import compiler

from pathlib import Path
import sys

# Add component locations
sys.path.append(str(Path(__file__).parent.parent / "components" / "data-download"))
from data_download.component import create_mnist_data

train_component_path = Path(__file__).parent.parent / "components" / "train-container"
train_mnist_vae_model = kfp.components.load_component_from_file(
    str(train_component_path / "component.yaml")
)
# Because we are using V1, cant use the python specification - The whole point of this
# compont is to trial thwe container specification
# from train_container.component import train_mnist_vae_model

sys.path.append(str(Path(__file__).parent.parent / "components" / "hp-tuner"))
from hp_tuner.component import hp_search




@kfp.v2.dsl.pipeline(
    name="mnist-pipeline",
    description="A pipeline to train a VAE on MNIST",
)
def mnist_pipeline(text: str):
    # The function is understood by the compiler because we have built the component?
    create_mnist_data_task = create_mnist_data()

    train_mnist_vae_model_task = train_mnist_vae_model(
       data_locn=create_mnist_data_task.outputs["output"], latent_dim=10, batch_size=10
    )

    # with kfp.dsl.ParallelFor([1, 2, 3, 4, 5]):
    hp_search_task = hp_search(
        epochs=10,
        steps_per_epoch=10,
        tuner_project_name="test",
        data=create_mnist_data_task.outputs["output"],
    )


# Compile the pipeline
# It is important to enable V2_Compatible here
compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(
    pipeline_func=mnist_pipeline, package_path="mnist_pipeline.yaml"
)

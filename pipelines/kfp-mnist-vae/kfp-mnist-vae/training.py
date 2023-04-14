import kfp
from kfp import compiler

from pathlib import Path
import sys

# Add component locations
sys.path.append(str(Path(__file__).parent.parent / "components" / "data-download"))
from data_download.component import create_mnist_data

sys.path.append(str(Path(__file__).parent.parent / "components" / "hp-tuner"))
from hp_tuner.component import hp_search


@kfp.dsl.pipeline(
    name="mnist-pipeline",
    description="A pipeline to train a VAE on MNIST",
)
def mnist_pipeline(text: str):
    # The function is understood by the compiler because we have built the component?
    create_mnist_data_task = create_mnist_data()

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

import kfp
from kfp import compiler

from pathlib import Path
import sys 

# Add component locations
sys.path.append(str(Path(__file__).parent.parent / 'components' / 'data-download'))
from data_download.component import create_mnist_data




@kfp.dsl.pipeline(
    name="mnist-pipeline",
    description="A pipeline to train a VAE on MNIST",
)
def mnist_pipeline(text: str):
    # The function is understood by the compiler because we have built the component?
    create_mnist_data_task = create_mnist_data()


# Compile the pipeline
# It is important to enable V2_Compatible here
compiler \
    .Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE) \
    .compile(
        pipeline_func=mnist_pipeline,
        package_path='mnist_pipeline.yaml'
    )



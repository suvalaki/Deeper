import kfp
from kfp.v2 import compiler
from kfp.v2 import dsl
from kfp.v2.dsl import  Output, Dataset, Artifact
from typing import *

@dsl.component(
    base_image="suvalaki/deeper:latest",
    target_image="suvalaki/deeper-pipeline-config_generator:latest",
    packages_to_install=['kfp'],
    # output yaml 
    output_component_file="component_create_config.yaml",
    #install_kfp_package=False,
)
def create_config(configuration: Output[Artifact]):

    import tensorflow as tf
    import tensorflow_addons as tfa
    from deeper.models.vae import Vae, MultipleObjectiveDimensions

    #%% Instantiate the model
    config = Vae.Config(
        input_dimensions=MultipleObjectiveDimensions(
            regression=0,
            # boolean=X_train.shape[-1],
            boolean=28,
            ordinal=(0,),
            categorical=(0,),
        ),
        output_dimensions=MultipleObjectiveDimensions(
            regression=0,
            # boolean=X_train.shape[-1],
            boolean=28,
            ordinal=(0,),
            categorical=(0,),
        ),
        encoder_embedding_dimensions=[512, 512, 256],
        decoder_embedding_dimensions=[512, 512, 256][::-1],
        latent_dim=64,
        embedding_activation=tf.keras.layers.Activation("elu"),
        kld_z_schedule=tfa.optimizers.CyclicalLearningRate(
            1.0, 1.0, step_size=30000.0, scale_fn=lambda x: 1.0, scale_mode="cycle"
        ),
    )

    with open(configuration.path, "w") as file:
        file.write(config.json(indent=4))

    print(config.json(indent=4))


#compiler.Compiler().compile(create_config, 'component_create_config.yaml')
import kfp
from kfp.v2 import compiler
from kfp.v2 import dsl
from kfp.v2.dsl import Input, Output, Dataset, Artifact
from typing import *


@dsl.component(
    base_image="suvalaki/deeper:latest",
    output_component_file="hp_tuner.yaml",
    packages_to_install=["keras_tuner"],
)
def hp_search(
    epochs: int,
    steps_per_epoch: int,
    tuner_project_name: str,
    data: Input[Dataset],
    hyper_parameter_storage: Output[Artifact],
    executions_per_trial: int = 2,
    max_trials: int = 20,
    number_best_hp_to_keep: int = 2,
):

    import logging
    import pickle

    import numpy as np
    import tensorflow as tf
    import tensorflow_addons as tfa
    import keras_tuner as kt

    from deeper.models.vae import Vae, MultipleObjectiveDimensions
    from deeper.optimizers.automl.tunable_types import (
        TunableActivation,
    )

    logger = logging.getLogger(__name__)

    DATA_DIM = 28 * 28
    BATCH_SIZE = 64
    # STRATEGY = tf.distribute.MirroredStrategy()

    def build_model(hp):

        config = Vae.Config(
            input_dimensions=MultipleObjectiveDimensions(
                regression=0,
                boolean=DATA_DIM,
                ordinal=(0,),
                categorical=(0,),
            ),
            output_dimensions=MultipleObjectiveDimensions(
                regression=0,
                boolean=DATA_DIM,
                ordinal=(0,),
                categorical=(0,),
            ),
            embedding_activation=TunableActivation("elu"),
        ).parse_tunable(hp)

        logger.info(config)
        model = Vae(config)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), run_eagerly=True)
        return model

    def load_data():
        with open(data.path, "rb") as file:
            d = pickle.load(file)
        return d

    X_train, X_test, y_train, y_test = load_data()
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, X_train)).batch(BATCH_SIZE)
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, X_test)).batch(BATCH_SIZE)

    tuner = kt.RandomSearch(
        build_model,
        objective=kt.Objective("val_losses/loss", direction="min"),  # max_trials=150,
        max_trials=max_trials,
        # distribution_strategy=STRATEGY,
        executions_per_trial=executions_per_trial,
        directory=hyper_parameter_storage.path,
        project_name=tuner_project_name,
    )

    logging.info("search space summary:")
    logging.info(tuner.search_space_summary())

    logging.info("hp tuning model....")
    tuner._display.col_width = 60  # so we can see parameters logged
    tuner.search(
        ds_train,
        validation_data=ds_test,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=0,
    )

    logging.info("Finished")

    # best_model = tuner.get_best_models()[0]

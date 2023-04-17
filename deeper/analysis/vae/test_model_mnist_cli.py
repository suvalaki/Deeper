import pydantic
import pydantic_argparse


class InputDataArguments(pydantic.BaseSettings):
    data_locn: str


# Simple model arguments
class InputModelArguments(pydantic.BaseSettings):
    latent_dim: int
    batch_size: int


class InputArguments(InputDataArguments, InputModelArguments):
    ...


def load_data(data_locn):
    import pickle

    with open(data_locn, "rb") as file:
        d = pickle.load(file)
    return d


def main():

    parser = pydantic_argparse.ArgumentParser(
        model=InputArguments,
        prog="VAE MNIST",
        description="Simple runner for vae mnist model",
        version="0.0.1",
    )

    args = parser.parse_typed_args()

    import tensorflow as tf
    import tensorflow_addons as tfa
    import numpy as np

    tf.config.set_visible_devices([], "GPU")

    from deeper.models.vae import Vae
    from deeper.models.vae import MultipleObjectiveDimensions

    # Load the data
    data = load_data(args.data_locn)
    X_train, X_test, y_train, y_test = data

    # Create model
    config = Vae.Config(
        input_dimensions=MultipleObjectiveDimensions(
            regression=0,
            boolean=X_train.shape[-1],
            ordinal=(0,),
            categorical=(0,),
        ),
        output_dimensions=MultipleObjectiveDimensions(
            regression=0,
            boolean=X_train.shape[-1],
            ordinal=(0,),
            categorical=(0,),
        ),
        encoder_embedding_dimensions=[512, 512, 256],
        decoder_embedding_dimensions=[512, 512, 256][::-1],
        latent_dim=args.latent_dim,
        embedding_activation=tf.keras.layers.Activation("elu"),
        kld_z_schedule=tfa.optimizers.CyclicalLearningRate(
            1.0, 1.0, step_size=30000.0, scale_fn=lambda x: 1.0, scale_mode="cycle"
        ),
    )
    model = Vae(config)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    # Train the model
    model.fit(
        X_train,
        X_train,
        epochs=10,
        # callbacks=[tbc, pc],
        batch_size=args.batch_size,
        validation_data=(X_test, X_test),
    )


if __name__ == "__main__":
    main()

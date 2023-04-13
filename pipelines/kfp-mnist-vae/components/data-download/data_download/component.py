import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import Output, Dataset
from typing import *

# This is a simple python component which doesnt need to create a new dockerfile 
# or build a new image. It just executes a python function using the base image.
# An alternative to using deeper:latest is just to pull in a default tensorflow
# image or to install slim/python3.10 and then specify tensorflow as a runtime 
# dependency in the component.yaml file (or using the keywork argumnent on the)
# decorator.

# The component can be tested by extracting the decorated function (from within
# the `python_func` attributte of the component) and then running alone; 
# For inputs a mock dsl.Input/dsl.Output object can be used to simulate the
# inputs and outputs of the component.

# There are some significant downsides to this approach: namely that testcases 
# cannot be written for functions defined within the component. 
# In the case that there are multiple functions we want to test then we would
# need to create a new component for each function. But this creates deployment
# complexity as we would deploy a new pod for each function; and we get all the 
# IO complexity in terms of passing data between the functions.


@dsl.component(
    base_image="suvalaki/deeper:latest",
    output_component_file="download_mnist_data.yaml",
)
def create_mnist_data(output: Output[Dataset]):
    """Load Mnist Data from the tensorflow package and save it as a 1d reshaped
    numpy float array"""

    import tensorflow as tf
    import numpy as np
    import pickle

    PIXELS_HIGH = 28
    PIXELS_WIDE = 28
    COLOR_RANGE = 256.0 - 1
    BLACK_WHITE_THRESHOLD = 0.5

    pixels_squared = PIXELS_HIGH * PIXELS_WIDE

    def reshape_to_1d(data: np.ndarray) -> np.ndarray:
        return data.reshape(data.shape[0], pixels_squared)

    def norm_color(data: np.ndarray) -> np.ndarray:
        return data / COLOR_RANGE

    def binarize(data: np.ndarray) -> np.ndarray:
        return (data > BLACK_WHITE_THRESHOLD).astype(float)

    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train, X_test = map(reshape_to_1d, (X_train, X_test))

    # Transform the data to be binary encoded
    X_train, X_test = map(norm_color, (X_train, X_test))
    X_train, X_test = map(binarize, (X_train, X_test))
    transformed_data = (X_train, X_test, y_train, y_test)

    with open(output.path, "wb") as file:
        pickle.dump(transformed_data, file)

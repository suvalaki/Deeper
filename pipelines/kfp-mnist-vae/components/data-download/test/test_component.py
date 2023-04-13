import unittest
import pickle

import numpy as np

from tempfile import TemporaryDirectory
from pathlib import Path

from data_download.component import create_mnist_data

# A simple example of how to produce some basic testcases 
# for a lightweight-python component. 
# The key is to Mock the output object and call the 
# underlying python function directly.

class MockOutput:

    """Simple wrapper to mimic the Output KFP object. 
    Uses a temporary directory to store the output data.
    Automatically cleaned up on destruction"""

    def __init__(self):
        self._tmpdir = TemporaryDirectory()
        self._path = Path(self._tmpdir.name) / "data.pkl"

    @property 
    def path(self):
        return str(self._path)


class Test_create_mnist_data(unittest.TestCase):


    def test_create_mnist_data(self):

        # Create a mock output
        output = MockOutput()

        # Run the function
        create_mnist_data.python_func(output)

        # Check that the output file exists
        self.assertTrue(Path(output.path).exists())

        # Check that all of the data is there
        with open(output.path, "rb") as file:
            data = pickle.load(file)

        self.assertEqual(len(data), 4)

        # Check that the data takes the right values,
        # between 0 and 1
        X_train, X_test, y_train, y_test = data
        self.assertTrue(np.all(X_train >= 0))
        self.assertTrue(np.all(X_train <= 1))



if __name__ == "__main__":
    unittest.main()
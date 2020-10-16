from deeper.models.gmvae.gmvae_pure_sampling import wrapper
import json
import numpy as np
import tensorflow as tf

CONFIG_FILE = "./configs/models/gmvae/gmvae_pure_sampling/defaultconfig.json"

def test_load_config():

    config_dict = json.loads(open(CONFIG_FILE).read())
    rand = np.random.RandomState(123)
    X = rand.random((100, config_dict["input_dimension"]))


    with tf.device("/cpu:0"):
        model = wrapper.ModelWrapper(config_dict)


    with tf.device("/cpu:0"):
        y = model(X)



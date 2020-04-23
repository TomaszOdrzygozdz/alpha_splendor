from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import gin
import numpy as np

from splendor.envs.mechanics.state import State
from splendor.networks.architectures.average_pooling_network import splendor_state_evaluator, GemsEncoder, PriceEncoder, \
    ManyCardsEncoder
from splendor.networks.utils.vectorizer import Vectorizer

gin.parse_config_file('/home/tomasz/ML_Research/alpha_splendor/own_testing/one_side_splendor/network_params.gin')

x = splendor_state_evaluator()
x.compile(loss='mse')

vectorizer = Vectorizer()
y = x.predict(vectorizer.state_to_input(State()))
print(y)
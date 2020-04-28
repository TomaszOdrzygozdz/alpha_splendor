from alpacka.data import nested_stack
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import gin
import numpy as np
from own_testing.one_side_splendor.fifo_input import xxx

from splendor.envs.mechanics.state import State
from splendor.networks.architectures.average_pooling_network import splendor_state_evaluator, GemsEncoder, PriceEncoder, \
    ManyCardsEncoder
from splendor.networks.utils.vectorizer import Vectorizer

gin.parse_config_file('/home/tomasz/ML_Research/alpha_splendor/own_testing/one_side_splendor/network_params.gin')

x = splendor_state_evaluator(None,)
x.compile(loss='mse')

vectorizer = Vectorizer()
obs = [vectorizer.state_to_input(State()) for _ in range(15)]
z = nested_stack(obs)
o = x.predict_on_batch(z)
#print(o)
for tt in z:
    print(tt.shape)

print(len(z))

print(z)

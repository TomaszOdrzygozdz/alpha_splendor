"""Test for signature method in utils"""

import numpy as np

from gym.spaces import Tuple, Discrete, Box
from alpacka.data import TensorSignature
from alpacka.utils.space import signature

def test_signature_for_tuples():
    """Test generation of signatures for observation that are gym Tuples"""
    observation_space = Tuple((Tuple((Box(np.array([-2, -2]), np.array([2, 2])),
                                      Discrete(3))), Tuple((Discrete(4), Discrete(5)))))
    observation_space_signature = signature(observation_space)
    assert observation_space_signature == ((TensorSignature(shape=(2,), dtype=np.dtype('float32')),
                                            TensorSignature(shape=(), dtype=np.dtype('int64'))),
                                           (TensorSignature(shape=(), dtype=np.dtype('int64')),
                                            TensorSignature(shape=(), dtype=np.dtype('int64'))))

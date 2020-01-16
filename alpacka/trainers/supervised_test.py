"""Tests for alpacka.trainers.supervised."""

import collections

import numpy as np

from alpacka import data
from alpacka.networks import keras
from alpacka.trainers import supervised


_TestTransition = collections.namedtuple('_TestTransition', ['observation'])


def test_integration_with_keras():
    # Just a smoke test, that nothing errors out.
    n_transitions = 10
    obs_shape = (4,)
    network_sig = data.NetworkSignature(
        input=data.TensorSignature(shape=obs_shape),
        output=data.TensorSignature(shape=(1,)),
    )
    trainer = supervised.SupervisedTrainer(
        network_signature=network_sig,
        target=supervised.target_solved,
        batch_size=2,
        n_steps_per_epoch=3,
        replay_buffer_capacity=n_transitions,
    )
    trainer.add_episode(data.Episode(
        transition_batch=_TestTransition(
            observation=np.zeros((n_transitions,) + obs_shape),
        ),
        return_=123,
        solved=False,
    ))
    network = keras.KerasNetwork(network_signature=network_sig)
    trainer.train_epoch(network)

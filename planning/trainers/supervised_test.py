"""Tests for planning.trainers.supervised."""

import collections

import numpy as np

from planning import data
from planning.networks import keras
from planning.trainers import supervised


_TestTransition = collections.namedtuple(
    '_TestTransition', ['observation', 'solved']
)


def test_integration_with_keras():
    # Just a smoke test, that nothing errors out.
    n_transitions = 10
    obs_shape = (4,)
    trainer = supervised.SupervisedTrainer(
        input_shape=obs_shape,
        target_fn=supervised.target_solved,
        batch_size=2,
        n_steps_per_epoch=3,
        replay_buffer_capacity=n_transitions,
    )
    trainer.add_episode(data.Episode(
        transition_batch=_TestTransition(
            observation=np.zeros((n_transitions,) + obs_shape),
            solved=np.zeros((n_transitions,), dtype=np.bool),
        ),
        return_=123,
    ))
    network = keras.KerasNetwork(input_shape=obs_shape)
    trainer.train_epoch(network)

"""Datatypes and functions for manipulating them."""

import collections

from planning.data.ops import *


# Transition between two states, S and S'.
Transition = collections.namedtuple(
    'Transition',
    [
        # Observation obtained at S.
        'observation',
        # Action played based on the observation.
        'action',
        # Reward obtained after performing the action.
        'reward',
        # Whether the environment is "done" at S'.
        'done',
        # Observation obtained at S'.
        'next_observation',
        # Whether the environment is "solved" at S'.
        'solved',
    ],
    defaults=(
        None,  # solved
    ),
)

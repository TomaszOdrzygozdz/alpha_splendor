"""Datatypes and functions for manipulating them."""

import collections

from planning.data.ops import nested_map, nested_zip, nested_unzip, nested_stack, nested_unstack, nested_concatenate  # noqa: F401, E501


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
    ]
)
Transition.__new__.__defaults__ = (None,)  # solved


# Basic Episode object, summarizing experience collected when solving an episode
# in the form of transitions. It's used for basic Agent -> Trainer
# communication. Agents and Trainers can use different (but shared) episode
# representations, as long as they have a 'return_' field, as this field is used
# by Runner for reporting metrics.
Episode = collections.namedtuple(
    'Episode',
    [
        # Transition object containing a batch of transitions.
        'transition_batch',
        # Undiscounted return (cumulative reward) for the entire episode.
        'return_',
    ]
)
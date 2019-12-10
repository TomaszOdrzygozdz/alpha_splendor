"""Tests for alpacka.agents.shooting."""

import numpy as np

from alpacka import agents
from alpacka import data
from alpacka.agents import shooting


def construct_episodes(actions, returns):
    episodes = []
    for acts, rets in zip(actions, returns):
        transition_batch = [
            data.Transition(None, act, ret, False, None)
            for act, ret in zip(acts[:-1], rets[:-1])]
        transition_batch.append(
            data.Transition(None, acts[-1], rets[-1], True, None))
        episodes.append(data.Episode(transition_batch, sum(rets)))
    return episodes


def test_mean_aggregate_episodes():
    # Set up
    actions = [[0], [1], [1], [2], [2], [2]]
    returns = [[-1], [0], [1], [1], [2], [-1]]
    gt_scores = np.array([-1, 1/2, 2/3])
    episodes = construct_episodes(actions, returns)

    # Run
    mean_scores = shooting.mean_aggregate(3, episodes)

    # Test
    np.testing.assert_array_equal(gt_scores, mean_scores)


def test_max_aggregate_episodes():
    # Set up
    actions = [[0], [1], [1], [2], [2], [2]]
    returns = [[-1], [0], [1], [1], [2], [-1]]
    gt_scores = np.array([-1, 1, 2])
    episodes = construct_episodes(actions, returns)

    # Run
    mean_scores = shooting.max_aggregate(3, episodes)

    # Test
    np.testing.assert_array_equal(gt_scores, mean_scores)

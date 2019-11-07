"""Tests for planning.agents.mcts."""

import asyncio
import functools

import gym
import pytest

from planning import agents
from planning import envs

import numpy as np


class TabularEnv(envs.ModelEnv):
    """Tabular environment with hardcoded transitions.

    Observations are equal to states.
    """

    def __init__(self, init_state, n_actions, transitions):
        """Initializes TabularEnv.

        Args:
            init_state: Initial state, returned from reset().
            transitions: Dict of structure:
                {
                    state: {
                        action: (state', reward, done),
                        # ...
                    },
                    # ...
                }
        """
        self.observation_space = gym.spaces.Discrete(len(transitions))
        self.action_space = gym.spaces.Discrete(n_actions)
        self._init_state = init_state
        self._transitions = transitions

    def reset(self):
        self._state = self._init_state
        return self._state

    def step(self, action):
        (self._state, reward, done) = self._transitions[self._state][action]
        return (self._state, reward, done, {})

    def clone_state(self):
        return self._state

    def restore_state(self, state):
        self._state = state


@asyncio.coroutine
def rate_new_leaves_tabular(
    leaf, observation, model, discount, child_qualities
):
    del leaf
    del model
    del discount
    return child_qualities[observation]


def run_without_suspensions(coroutine):
    try:
        next(coroutine)
        assert False, 'Coroutine should return immediately.'
    except StopIteration as e:
        return e.value


def test_integration_with_cartpole():
    env = envs.CartPole()
    agent = agents.MCTSAgent(
        n_passes=2,
        rate_new_leaves_fn=functools.partial(
            agents.mcts.rate_new_leaves_with_rollouts,
            rollout_time_limit=2,
        )
    )
    transition_batch = run_without_suspensions(agent.solve(env))
    assert transition_batch.observation.shape[0] > 0


def test_act_doesnt_change_env_state():
    env = envs.CartPole()
    agent = agents.MCTSAgent(
        n_passes=2,
        rate_new_leaves_fn=functools.partial(
            agents.mcts.rate_new_leaves_with_rollouts,
            rollout_time_limit=2,
        )
    )
    agent.reset(env)
    observation = env.reset()

    state_before = env.clone_state()
    run_without_suspensions(agent.act(observation))
    state_after = env.clone_state()
    np.testing.assert_equal(state_before, state_after)


def make_one_level_binary_tree(
    left_quality, right_quality, left_reward=0, right_reward=0
):
    # 0, action 0 -> 1 (left)
    # 0, action 1 -> 2 (right)
    (root_state, left_state, right_state) = (0, 1, 2)
    env = TabularEnv(
        init_state=root_state,
        n_actions=2,
        transitions={
            # state: {action: (state', reward, done)}
            root_state: {
                0: (left_state, left_reward, False),
                1: (right_state, right_reward, False),
            },
            # Dummy terminal states, made so we can expand left and right.
            left_state: {0: (3, 0, True), 1: (4, 0, True)},
            right_state: {0: (5, 0, True), 1: (6, 0, True),},
        }
    )
    rate_new_leaves_fn = functools.partial(
        rate_new_leaves_tabular,
        child_qualities={
            root_state: [left_quality, right_quality],
            left_state: [0, 0],
            right_state: [0, 0],
        },
    )
    return (env, rate_new_leaves_fn)


@pytest.mark.parametrize(
    "left_quality,right_quality,expected_action", [
        (1, 0, 0),  # Should choose left because of high quality.
        (0, 1, 1),  # Should choose right because of high quality.
    ]
)
def test_decision_after_one_pass(left_quality, right_quality, expected_action):
    # 0, action 0 -> 1 (left)
    # 0, action 1 -> 2 (right)
    # 1 pass, should choose depending on qualities.
    (env, rate_new_leaves_fn) = make_one_level_binary_tree(
        left_quality, right_quality
    )
    agent = agents.MCTSAgent(n_passes=1, rate_new_leaves_fn=rate_new_leaves_fn)
    agent.reset(env)
    observation = env.reset()
    actual_action = run_without_suspensions(agent.act(observation))
    assert actual_action == expected_action

def test_stops_on_done():
    # 0 -> 1 (done)
    # 2 passes, env is not stepped from 1.
    env = TabularEnv(
        init_state=0,
        n_actions=1,
        transitions={0: {0: (1, 0, True)}},
    )
    agent = agents.MCTSAgent(
        n_passes=2,
        rate_new_leaves_fn=functools.partial(
            rate_new_leaves_tabular,
            child_qualities={0: [0]},
        )
    )
    agent.reset(env)
    observation = env.reset()
    # rate_new_leaves_fn errors out when rating children of a state not in the
    # quality table.
    run_without_suspensions(agent.act(observation))


def test_backtracks_because_of_quality():
    # 0, action 0 -> 1 (medium quality)
    # 0, action 1 -> 2 (high quality)
    # 2, action 0 -> 3 (very low quality)
    # 2, action 1 -> 3 (very low quality)
    # 3 passes, should choose 0.
    env = TabularEnv(
        init_state=0,
        n_actions=2,
        transitions={
            # Root.
            0: {0: (1, 0, False), 1: (2, 0, False)},
            # Left branch, ending here.
            1: {0: (3, 0, True), 1: (4, 0, True)},
            # Right branch, one more level.
            2: {0: (5, 0, False), 1: (6, 0, False)},
            # End of the right branch.
            5: {0: (7, 0, True), 1: (8, 0, True)},
            6: {0: (9, 0, True), 1: (10, 0, True)},
        },
    )
    agent = agents.MCTSAgent(
        n_passes=3,
        rate_new_leaves_fn=functools.partial(
            rate_new_leaves_tabular,
            child_qualities={
                0: [0, 1],
                1: [0, 0],
                2: [-10, -10],
                5: [0, 0],
                6: [0, 0],
            },
        )
    )
    agent.reset(env)
    observation = env.reset()
    action = run_without_suspensions(agent.act(observation))
    assert action == 0


def test_backtracks_because_of_reward():
    # 0, action 0 -> 1 (high quality, very low reward)
    # 0, action 1 -> 2 (medium quality)
    # 2 passes, should choose 1.
    (env, rate_new_leaves_fn) = make_one_level_binary_tree(
        left_quality=1, left_reward=-10, right_quality=0, right_reward=0
    )
    agent = agents.MCTSAgent(
        n_passes=2,
        rate_new_leaves_fn=rate_new_leaves_fn,
    )
    agent.reset(env)
    observation = env.reset()
    action = run_without_suspensions(agent.act(observation))
    assert action == 1

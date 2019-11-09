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
    leaf, observation, model, discount, state_values
):
    del leaf
    del observation
    init_state = model.clone_state()
    def quality(action):
        (observation, reward, _, _) = model.step(action)
        model.restore_state(init_state)
        # State is the same as observation.
        return reward + discount * state_values[observation]
    return [quality(action) for action in range(model.action_space.n)]


def run_without_suspensions(coroutine):
    try:
        next(coroutine)
        assert False, 'Coroutine should return immediately.'
    except StopIteration as e:
        return e.value


@pytest.mark.parametrize('graph_mode', [False, True])
def test_integration_with_cartpole(graph_mode):
    env = envs.CartPole()
    agent = agents.MCTSAgent(
        n_passes=2,
        rate_new_leaves_fn=functools.partial(
            agents.mcts.rate_new_leaves_with_rollouts,
            rollout_time_limit=2,
        ),
        graph_mode=graph_mode,
    )
    transition_batch = run_without_suspensions(agent.solve(env))
    assert transition_batch.observation.shape[0] > 0


@pytest.mark.parametrize('graph_mode', [False, True])
def test_act_doesnt_change_env_state(graph_mode):
    env = envs.CartPole()
    agent = agents.MCTSAgent(
        n_passes=2,
        rate_new_leaves_fn=functools.partial(
            agents.mcts.rate_new_leaves_with_rollouts,
            rollout_time_limit=2,
        ),
        graph_mode=graph_mode,
    )
    agent.reset(env)
    observation = env.reset()

    state_before = env.clone_state()
    run_without_suspensions(agent.act(observation))
    state_after = env.clone_state()
    np.testing.assert_equal(state_before, state_after)


def make_one_level_binary_tree(
    left_value, right_value, left_reward=0, right_reward=0
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
        state_values={
            root_state: 0,
            left_state: left_value,
            right_state: right_value,
            # Dummy terminal states.
            3: 0, 4: 0, 5: 0, 6: 0,
        },
    )
    return (env, rate_new_leaves_fn)


@pytest.mark.parametrize(
    "left_value,right_value,expected_action", [
        (1, 0, 0),  # Should choose left because of high value.
        (0, 1, 1),  # Should choose right because of high value.
    ]
)
@pytest.mark.parametrize('graph_mode', [False, True])
def test_decision_after_one_pass(
    left_value, right_value, expected_action, graph_mode
):
    # 0, action 0 -> 1 (left)
    # 0, action 1 -> 2 (right)
    # 1 pass, should choose depending on qualities.
    (env, rate_new_leaves_fn) = make_one_level_binary_tree(
        left_value, right_value
    )
    agent = agents.MCTSAgent(
        n_passes=1,
        rate_new_leaves_fn=rate_new_leaves_fn,
        graph_mode=graph_mode,
    )
    agent.reset(env)
    observation = env.reset()
    actual_action = run_without_suspensions(agent.act(observation))
    assert actual_action == expected_action


@pytest.mark.parametrize('graph_mode', [False, True])
def test_stops_on_done(graph_mode):
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
            state_values={0: 0, 1: 0},
        ),
        graph_mode=graph_mode,
    )
    agent.reset(env)
    observation = env.reset()
    # rate_new_leaves_fn errors out when rating nodes not in the value table.
    run_without_suspensions(agent.act(observation))


@pytest.mark.parametrize('graph_mode', [False, True])
def test_backtracks_because_of_value(graph_mode):
    # 0, action 0 -> 1 (medium value)
    # 0, action 1 -> 2 (high value)
    # 2, action 0 -> 3 (very low value)
    # 2, action 1 -> 3 (very low value)
    # 2 passes, should choose 0.
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
        n_passes=2,
        rate_new_leaves_fn=functools.partial(
            rate_new_leaves_tabular,
            state_values={
                0: 0,
                1: 0,
                2: 1,
                5: -10,
                6: -10,
            },
        ),
        graph_mode=graph_mode,
    )
    agent.reset(env)
    observation = env.reset()
    action = run_without_suspensions(agent.act(observation))
    assert action == 0


@pytest.mark.parametrize('graph_mode', [False, True])
def test_backtracks_because_of_reward(graph_mode):
    # 0, action 0 -> 1 (high value, very low reward)
    # 0, action 1 -> 2 (medium value)
    # 2 passes, should choose 1.
    (env, rate_new_leaves_fn) = make_one_level_binary_tree(
        left_value=1, left_reward=-10, right_value=0, right_reward=0
    )
    agent = agents.MCTSAgent(
        n_passes=2,
        rate_new_leaves_fn=rate_new_leaves_fn,
        graph_mode=graph_mode,
    )
    agent.reset(env)
    observation = env.reset()
    action = run_without_suspensions(agent.act(observation))
    assert action == 1


@pytest.mark.parametrize(
    'graph_mode,expected_second_action', [(True, 1), (False, 0)]
)
def test_caches_values_in_graph_mode(graph_mode, expected_second_action):
    # 0, action 0 -> 1 (high value)
    # 1, action 0 -> 2 (very low value)
    # 1, action 1 -> 3 (medium value)
    # 3, action 0 -> 4 (very low value)
    # 3, action 1 -> 5 (very low value)
    # 0, action 1 -> 6 (medium value)
    # 6, action 0 -> 1 (high value)
    # 6, action 1 -> 7 (medium value)
    # 3 passes for the first and 2 for the second action. In graph mode, should
    # choose 1, then 1. Not in graph mode, should choose 1, then 0.
    env = TabularEnv(
        init_state=0,
        n_actions=2,
        transitions={
            # Root.
            0: {0: (1, 0, False), 1: (6, 0, False)},
            # Left branch, long one.
            1: {0: (2, 0, True), 1: (3, 0, False)},
            3: {0: (4, 0, True), 1: (5, 0, True)},
            # Right branch, short, with a connection to the left.
            6: {0: (1, 0, False), 1: (7, 0, True)},
        },
    )
    agent = agents.MCTSAgent(
        n_passes=3,
        rate_new_leaves_fn=functools.partial(
            rate_new_leaves_tabular,
            state_values={
                0: 0,
                1: 1,
                2: -10,
                3: 0,
                4: -10,
                5: -10,
                6: 0,
                7: 0,
            },
        ),
        graph_mode=graph_mode,
    )
    agent.reset(env)

    observation = env.reset()
    first_action = run_without_suspensions(agent.act(observation))
    assert first_action == 1

    agent.n_passes = 2
    (observation, _, _, _) = env.step(first_action)
    second_action = run_without_suspensions(agent.act(observation))
    assert second_action == expected_second_action

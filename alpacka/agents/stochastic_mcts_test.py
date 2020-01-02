"""Tests for alpacka.agents.stochastic_mcts."""

import asyncio
import functools

import numpy as np
import pytest

from alpacka import agents
from alpacka import envs
from alpacka import testing


@asyncio.coroutine
def rate_new_leaves_tabular(
    leaf, observation, model, discount, state_values
):
    """Rates new leaves based on hardcoded values."""
    del leaf
    del observation
    init_state = model.clone_state()

    def quality(action):
        (observation, reward, _, _) = model.step(action)
        model.restore_state(init_state)
        # State is the same as observation.
        return reward + discount * state_values[observation]

    return [quality(action) for action in range(model.action_space.n)]


def test_integration_with_cartpole():
    env = envs.CartPole()
    agent = agents.StochasticMCTSAgent(
        n_passes=2,
        rate_new_leaves_fn=functools.partial(
            agents.stochastic_mcts.rate_new_leaves_with_rollouts,
            rollout_time_limit=2,
        ),
    )
    episode = testing.run_without_suspensions(agent.solve(env))
    assert episode.transition_batch.observation.shape[0]  # pylint: disable=no-member


@pytest.mark.parametrize('rate_new_leaves_fn', [
    functools.partial(
        agents.stochastic_mcts.rate_new_leaves_with_rollouts,
        rollout_time_limit=2,
    ),
    agents.stochastic_mcts.rate_new_leaves_with_value_network,
])
def test_act_doesnt_change_env_state(rate_new_leaves_fn):
    env = envs.CartPole()
    agent = agents.StochasticMCTSAgent(
        n_passes=2,
        rate_new_leaves_fn=rate_new_leaves_fn,
    )
    observation = env.reset()
    testing.run_without_suspensions(agent.reset(env, observation))

    state_before = env.clone_state()
    testing.run_with_dummy_network(agent.act(observation))
    state_after = env.clone_state()
    np.testing.assert_equal(state_before, state_after)


def make_one_level_binary_tree(
    left_value, right_value, left_reward=0, right_reward=0
):
    """Makes a TabularEnv and rate_new_leaves_fn for a 1-level binary tree."""
    # 0, action 0 -> 1 (left)
    # 0, action 1 -> 2 (right)
    (root_state, left_state, right_state) = (0, 1, 2)
    env = testing.TabularEnv(
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
            right_state: {0: (5, 0, True), 1: (6, 0, True)},
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
    'left_value,right_value,left_reward,right_reward,expected_action', [
        (1, 0, 0, 0, 0),  # Should choose left because of high value.
        (0, 1, 0, 0, 1),  # Should choose right because of high value.
        (0, 0, 1, 0, 0),  # Should choose left because of high reward.
        (0, 0, 0, 1, 1),  # Should choose right because of high reward.
    ]
)
def test_decision_after_one_pass(
    left_value,
    right_value,
    left_reward,
    right_reward,
    expected_action,
):
    # 0, action 0 -> 1 (left)
    # 0, action 1 -> 2 (right)
    # 1 pass, should choose depending on qualities.
    (env, rate_new_leaves_fn) = make_one_level_binary_tree(
        left_value, right_value, left_reward, right_reward
    )
    agent = agents.StochasticMCTSAgent(
        n_passes=1,
        rate_new_leaves_fn=rate_new_leaves_fn,
    )
    observation = env.reset()
    testing.run_without_suspensions(agent.reset(env, observation))
    (actual_action, _) = testing.run_without_suspensions(agent.act(observation))
    assert actual_action == expected_action


def test_stops_on_done():
    # 0 -> 1 (done)
    # 2 passes, env is not stepped from 1.
    env = testing.TabularEnv(
        init_state=0,
        n_actions=1,
        transitions={0: {0: (1, 0, True)}},
    )
    agent = agents.StochasticMCTSAgent(
        n_passes=2,
        rate_new_leaves_fn=functools.partial(
            rate_new_leaves_tabular,
            state_values={0: 0, 1: 0},
        ),
    )
    observation = env.reset()
    testing.run_without_suspensions(agent.reset(env, observation))
    # rate_new_leaves_fn errors out when rating nodes not in the value table.
    testing.run_without_suspensions(agent.act(observation))


def test_backtracks_because_of_value():
    # 0, action 0 -> 1 (medium value)
    # 0, action 1 -> 2 (high value)
    # 2, action 0 -> 3 (very low value)
    # 2, action 1 -> 3 (very low value)
    # 2 passes, should choose 0.
    env = testing.TabularEnv(
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
    agent = agents.StochasticMCTSAgent(
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
    )
    observation = env.reset()
    testing.run_without_suspensions(agent.reset(env, observation))
    (action, _) = testing.run_without_suspensions(agent.act(observation))
    assert action == 0


def test_backtracks_because_of_reward():
    # 0, action 0 -> 1 (high value, very low reward)
    # 0, action 1 -> 2 (medium value)
    # 2 passes, should choose 1.
    (env, rate_new_leaves_fn) = make_one_level_binary_tree(
        left_value=1, left_reward=-10, right_value=0, right_reward=0
    )
    agent = agents.StochasticMCTSAgent(
        n_passes=2,
        rate_new_leaves_fn=rate_new_leaves_fn,
    )
    observation = env.reset()
    testing.run_without_suspensions(agent.reset(env, observation))
    (action, _) = testing.run_without_suspensions(agent.act(observation))
    assert action == 1

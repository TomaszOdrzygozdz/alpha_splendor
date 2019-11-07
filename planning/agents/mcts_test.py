"""Tests for planning.agents.mcts."""

import functools

from planning import agents
from planning import envs


def test_integration_with_cartpole():
    env = envs.CartPole()
    agent = agents.MCTSAgent(
        n_passes=2,
        rate_new_leaves_fn=functools.partial(
            agents.mcts.rate_new_leaves_with_rollouts,
            rollout_time_limit=2,
        )
    )
    try:
        next(agent.solve(env))
        # Should return immediately.
        assert False
    except StopIteration as e:
        transition_batch = e.value
    assert transition_batch.observation.shape[0] > 0

"""Core agents."""

import asyncio

from planning.agents import base


class RandomAgent(base.OnlineAgent):
    """Random agent, sampling actions from the uniform distribution."""

    def solve(self, env):
        self._action_space = env.action_space
        return super().solve(env)

    @asyncio.coroutine
    def act(self, observation):
        del observation
        return self._action_space.sample()

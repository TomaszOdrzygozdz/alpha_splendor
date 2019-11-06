"""Core agents."""

import asyncio

from planning.agents import base


class RandomAgent(base.OnlineAgent):
    """Random agent, sampling actions from the uniform distribution."""

    def __init__(self, action_space, **kwargs):
        super().__init__(action_space, **kwargs)
        self._action_space = action_space

    @asyncio.coroutine
    def act(self, observation):
        del observation
        return self._action_space.sample()

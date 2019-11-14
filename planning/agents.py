"""Agents."""

import asyncio

from planning import data
from planning import envs


class Agent:
    """Agent base class.

    Agents can use neural networks internally. Network prediction is run outside
    of the Agent, so it can be batched across multiple Agents for efficiency.
    This is done using a coroutine API, explained in solve().
    """

    def __init__(self):
        """No-op constructor just for documentation purposes."""
        pass

    def solve(self, env):
        """Solves a given environment.

        Coroutine, suspends execution for every neural network prediction
        request. This enables a very convenient interface for requesting
        predictions by the Agent:

            def solve(self, env):
                # Planning...
                predictions = yield inputs
                # Planning...
                predictions = yield inputs
                # Planning...
                return episode

        Example usage:

            coroutine = agent.solve(env)
            try:
                prediction_request = next(coroutine)
                network_output = process_request(prediction_request)
                prediction_request = coroutine.send(network_output)
                # Possibly more prediction requests...
            except StopIteration as e:
                episode = e.value

        Agents that do not use neural networks should wrap their solve() method
        in an @asyncio.coroutine decorator, so Python knows to treat it as
        a coroutine even though it doesn't have any yield.

        Args:
            env: (gym.Env) Environment to solve.

        Yields:
            A stream of Network inputs requested for inference.

        Returns:
            (Agent/Trainer-specific) Episode object summarizing the collected
            data for training the Network.
        """
        raise NotImplementedError


class OnlineAgent(Agent):
    """Base class for online agents, i.e. planning on a per-action basis.

    Provides a default implementation of Agent.solve(), returning a Transition
    object with the collected batch of transitions.
    """

    def act(self, observation):
        """Determines the next action to be performed.

        Coroutine, suspends execution similarly to Agent.solve().

        In model-based agents, the original environment state MUST be restored
        in the end of act(). This is not checked at runtime, since it would be
        a big overhead for heavier environments.

        Args:
            observation: Observation from the environment.

        Yields:
            A stream of Network inputs requested for inference.

        Returns:
            Action to make in the environment.
        """
        raise NotImplementedError

    def solve(self, env):
        """Solves a given environment using OnlineAgent.act().

        Args:
            env: (gym.Env) Environment to solve.

        Yields:
            A stream of Network inputs requested for inference.

        Returns:
            Transition object containing a batch of collected transitions.
        """
        # Wrap the environment in a wrapper for collecting transitions. Collection
        # is turned on/off for the Agent.act() to collect only transitions on
        # the real environment.
        self._env = envs.TransitionCollectorWrapper(env)

        self._env.collect = True
        observation = self._env.reset()
        done = False
        while not done:
            self._env.collect = False
            # Forward network prediction requests to BatchStepper.
            action = yield from self.act(observation)
            self._env.collect = True
            (observation, _, done, _) = self._env.step(action)

        return data.nested_stack(self._env.transitions)


class RandomAgent(OnlineAgent):
    """Random agent, sampling actions from the uniform distribution."""

    def solve(self, env):
        self._action_space = env.action_space
        return super().solve(env)

    @asyncio.coroutine
    def act(self, observation):
        del observation
        return self._action_space.sample()

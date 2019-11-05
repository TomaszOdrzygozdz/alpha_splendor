"""Agents."""

from planning import data
from planning import envs
from planning.data import messages


class Agent:
    """Agent base class.

    Agents can be either model-free or model-based, as they take the real
    environment as an argument in __init__ ("true model").

    Agents can use neural networks internally. Network prediction is run outside
    of the Agent, so it can be batched across multiple Agents for efficiency.
    This is done using a coroutine API, explained in act().
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
                predictions = yield messages.PredictRequest(inputs)
                # Planning...
                predictions = yield messages.PredictRequest(inputs)
                # Planning...
                yield messages.Episode(episode)

        Example usage:

            coroutine = agent.solve(env)
            prediction_request = next(coroutine)
            network_output = process_request(prediction_request)
            prediction_request = coroutine.send(network_output)
            network_output = process_request(prediction_request)
            # Possibly more prediction requests...
            episode = coroutine.send(network_output)

        Args:
            env: (gym.Env) Environment to solve.

        Yields:
            A stream of PredictMessages (requests for inference using the
            network), followed by an EpisodeMessage (data collected in the
            episode).
        """
        raise NotImplementedError


class OnlineAgent(Agent):
    """Base class for online agents, i.e. planning on a per-action basis."""

    def __init__(self, collect_real=False):
        self._collect_real = collect_real

    def act(self, observation):
        """Determines the next action to be performed.

        Coroutine, suspends execution similarly to Agent.solve().

        In model-based agents, the original environment state MUST be restored
        in the end of act(). This is not checked at runtime, since it would be
        a big overhead for heavier environments.

        Args:
            observation: Observation from the environment.

        Yields:
            A stream of PredictMessages (requests for inference using the
            network), followed by an ActionMessage (final action to be
            performed).
        """
        raise NotImplementedError

    def solve(self, env):
        """Solves a given environment using OnlineAgent.act()."""
        # Wrap the environment in a wrapper for collecting transitions. Collection
        # is turned on/off for the Agent.act() call based on collect_real.
        self._env = envs.TransitionCollectorWrapper(env)

        self._env.collect = self._collect_real
        observation = self._env.reset()
        done = False
        while not done:
            # Forward network prediction requests to BatchStepper.
            self._env.collect = not self._collect_real
            act_cor = self.act(observation)
            message = next(act_cor)
            while isinstance(message, messages.PredictRequest):
                prediction = yield message
                message = act_cor.send(prediction)

            # Once an action has been determined, run it on the environment.
            assert isinstance(message, messages.Action)
            self._env.collect = self._collect_real
            (observation, _, done, _) = self._env.step(message.action)

        yield messages.Episode(data.nested_stack(self._env.transitions))


class RandomAgent(OnlineAgent):
    """Random agent, sampling actions from the uniform distribution."""

    def solve(self, env):
        self._action_space = env.action_space
        return super().solve(env)

    def act(self, observation):
        del observation
        yield messages.Action(self._action_space.sample())

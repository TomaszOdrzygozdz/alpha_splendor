"""Agents."""

from planning.data import messages


class Agent:
    """Agent base class.

    Agents can be either model-free or model-based, as they take the real
    environment as an argument in __init__ ("true model").

    Agents can use neural networks internally. Network prediction is run outside
    of the Agent, so it can be batched across multiple Agents for efficiency.
    This is done using a coroutine API, explained in act().
    """

    def __init__(self, env):
        """No-op constructor just for documentation purposes."""
        del env

    def act(self, observation):
        """Determines the next action to be performed.

        Coroutine, suspends execution for every neural network prediction
        request. This enables a very convenient interface for requesting
        predictions by the Agent:
            
            def act(self, observation):
                # Planning...
                predictions = yield messages.PredictRequest(inputs)
                # Planning...
                predictions = yield messages.PredictRequest(inputs)
                # Planning...
                yield messages.Action(action)

        Example usage:

            coroutine = agent.act(observation)
            prediction_request = next(coroutine)
            network_output = process_request(prediction_request)
            prediction_request = coroutine.send(network_output)
            network_output = process_request(prediction_request)
            # Possibly more prediction requests...
            action = coroutine.send(network_output)

        In model-based agents, the original environment state MUST be restored
        in the end of act(). This is not checked at runtime, since it would be
        a big overhead for heavier environments.

        Yields:
            A stream of PredictMessages (requests for inference using the
            network), followed by an ActionMessage (final action to be
            performed).
        """
        raise NotImplementedError


class RandomAgent(Agent):
    """Random agent, sampling actions from the uniform distribution."""

    def __init__(self, env):
        super().__init__(env)
        self._action_space = env.action_space

    def act(self, observation):
        del observation
        yield messages.Action(self._action_space.sample())

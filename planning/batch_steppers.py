"""Environment steppers."""

from planning import data
from planning import envs
from planning.data import messages

import numpy as np


class BatchStepper:
    """Base class for running a batch of steppers.
    
    Abstracts out local/remote prediction using a Network.
    """

    def __init__(
        self, env_class, agent_class, network_class, n_envs, collect_real
    ):
        """No-op constructor just for documentation purposes."""
        del env_class
        del agent_class
        del network_class
        del n_envs
        del collect_real

    def run_episode_batch(self, params):
        """Runs a batch of episodes using the given network parameters.

        Args:
            params: (Network-dependent) Network parameters.

        Returns:
            Transition object with a batch of collected transitions.
        """
        raise NotImplementedError


def run_episode(env, agent, collect_real):
    """Runs a single episode on the environment.

    Uses a similar coroutine API as Agent.act() to forward prediction requests
    to the BatchStepper.

    Args:
        env: Environment.
        agent: Agent.
        collect_real: Whether to collect transition on the real environment or
            on the model.

    Yields:
        A stream of messages.PredictRequests (requests for inference using the
        network) followed by a messages.TransitionBatch. Reads network
        predictions back from the coroutine.
    """
    # Wrap the environment in a wrapper for collecting transitions. Collection
    # is turned on/off for the Agent.act() call based on collect_real.
    env = envs.TransitionCollectorWrapper(env)

    env.collect = collect_real
    observation = env.reset()
    done = False
    while not done:
        # Forward network prediction requests to BatchStepper.
        env.collect = not collect_real
        act_cor = agent.act(observation)
        message = next(act_cor)
        while isinstance(message, messages.PredictRequest):
            def assert_not_scalar(x):
                assert np.array(x).shape, (
                    'All arrays in a PredictRequest must be at least rank 1.'
                )
            data.nested_map(assert_not_scalar, message)

            prediction = yield message
            message = act_cor.send(prediction)

        # Once an action has been determined, run it on the environment.
        assert isinstance(message, messages.Action)
        env.collect = collect_real
        (observation, _, done, _) = env.step(message.action)

    yield messages.TransitionBatch(data.nested_stack(env.transitions))


class LocalBatchStepper(BatchStepper):
    """Batch stepper running locally.
    
    Runs batched prediction for all Agents at the same time.
    """

    def __init__(
        self, env_class, agent_class, network_class, n_envs, collect_real
    ):
        super().__init__(
            env_class, agent_class, network_class, n_envs, collect_real
        )
        def make_env_and_agent():
            env = env_class()
            return (env, agent_class(env))
        self._envs_and_agents = [make_env_and_agent() for _ in range(n_envs)]
        self._network = network_class()
        self._collect_real = collect_real

    def _batch_coroutines(self, cors):
        """Batches a list of coroutines into one.

        Handles waiting for the slowest coroutine and filling blanks in
        prediction requests.
        """
        # Store the final transitions in a list.
        transition_batches = [None] * len(cors)
        def store_transitions(i, cor):
            message = next(cor)
            while True:
                if isinstance(message, messages.TransitionBatch):
                    transition_batches[i] = message
                    break
                else:
                    prediction = yield message
                    message = cor.send(prediction)
            # End with an infinite stream of Nones, so we don't have
            # to deal with StopIteration later on.
            while True:
                yield None
        cors = [store_transitions(i, cor) for(i, cor) in enumerate(cors)]

        def batch_requests(xs):
            assert xs

            if all(x is None for x in xs):
                # All coroutines have finished - return the final transitions.
                return data.nested_concatenate(transition_batches)

            # PredictRequest used as a filler for coroutines that have already
            # finished.
            filler = next(x for x in xs if x is not None)
            # Fill with 0s for easier debugging.
            filler = data.nested_map(np.zeros_like, filler)
            assert isinstance(filler, messages.PredictRequest)

            # Substitute the filler for Nones.
            xs = [x if x is not None else filler for x in xs]

            # Stack instead of concatenate to ensure that all requests have
            # the same shape.
            x = data.nested_stack(xs)
            def flatten_first_2_dims(x):
                return np.reshape(x, (-1,) + x.shape[2:])
            # (n_agents, n_requests, ...) -> (n_agents * n_requests, ...)
            return data.nested_map(flatten_first_2_dims, x)

        def unbatch_responses(x):
            def unflatten_first_2_dims(x):
                return np.reshape(
                    x, (len(self._envs_and_agents), -1) + x.shape[1:]
                )
            # (n_agents * n_requests, ...) -> (n_agents, n_requests, ...)
            return data.nested_unstack(
                data.nested_map(unflatten_first_2_dims, x)
            )

        inputs = yield batch_requests([
            next(cor) for (i, cor) in enumerate(cors)
        ])
        while True:
            inputs = yield batch_requests([
                cor.send(inp)
                for (i, (cor, inp)) in enumerate(
                    zip(cors, unbatch_responses(inputs))
                )
            ])

    def run_episode_batch(self, params):
        self._network.params = params
        episode_cor = self._batch_coroutines([
            run_episode(env, agent, self._collect_real)
            for (env, agent) in self._envs_and_agents
        ])
        message = next(episode_cor)
        while isinstance(message, messages.PredictRequest):
            predictions = self._network.predict(message.inputs)
            message = episode_cor.send(predictions)

        assert isinstance(message, messages.TransitionBatch)
        return message.transition_batch

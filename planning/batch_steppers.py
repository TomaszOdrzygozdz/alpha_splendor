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
            List of completed episodes (Agent/Trainer-dependent).
        """
        raise NotImplementedError


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
        self._envs_and_agents = [
            (env_class(), agent_class()) for _ in range(n_envs)
        ]
        self._network = network_class()
        self._collect_real = collect_real

    def _batch_coroutines(self, cors):
        """Batches a list of coroutines into one.

        Handles waiting for the slowest coroutine and filling blanks in
        prediction requests.
        """
        # Store the final episodes in a list.
        episodes = [None] * len(cors)
        def store_transitions(i, cor):
            message = next(cor)
            while True:
                if isinstance(message, messages.Episode):
                    episodes[i] = message.episode
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
                # All coroutines have finished - return the final episodes.
                return messages.Episode(episodes)

            # PredictRequest used as a filler for coroutines that have already
            # finished.
            filler = next(x for x in xs if x is not None)
            # Fill with 0s for easier debugging.
            filler = data.nested_map(np.zeros_like, filler)
            assert isinstance(filler, messages.PredictRequest)

            # Substitute the filler for Nones.
            xs = [x if x is not None else filler for x in xs]

            def assert_not_scalar(x):
                assert np.array(x).shape, (
                    'All arrays in a PredictRequest must be at least rank 1.'
                )
            data.nested_map(assert_not_scalar, xs)

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
            agent.solve(env)
            for (env, agent) in self._envs_and_agents
        ])
        message = next(episode_cor)
        while isinstance(message, messages.PredictRequest):
            predictions = self._network.predict(message.inputs)
            message = episode_cor.send(predictions)

        # Return a list of completed episodes.
        assert isinstance(message, messages.Episode)
        return message.episode

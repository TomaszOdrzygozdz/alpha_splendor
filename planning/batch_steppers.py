"""Environment steppers."""

import gin
import numpy as np

from planning import data


class BatchStepper:
    """Base class for running a batch of steppers.

    Abstracts out local/remote prediction using a Network.
    """

    def __init__(
        self, env_class, agent_class, network_fn, n_envs
    ):
        """No-op constructor just for documentation purposes.

        Args:
            env_class (type): Environment class.
            agent_class (type): Agent class.
            network_fn (callable): Function () -> Network. Note: we take this
                instead of an already-initialized Network, because some
                BatchSteppers will send it to remote workers and it makes no
                sense to force Networks to be picklable just for this purpose.
            n_envs (int): Number of parallel environments to run.
        """
        del env_class
        del agent_class
        del network_fn
        del n_envs

    def run_episode_batch(self, params):
        """Runs a batch of episodes using the given network parameters.

        Args:
            params (Network-dependent): Network parameters.

        Returns:
            List of completed episodes (Agent/Trainer-dependent).
        """
        raise NotImplementedError


@gin.configurable
class LocalBatchStepper(BatchStepper):
    """Batch stepper running locally.

    Runs batched prediction for all Agents at the same time.
    """

    def __init__(self, env_class, agent_class, network_fn, n_envs):
        super().__init__(env_class, agent_class, network_fn, n_envs)

        def make_env_and_agent():
            env = env_class()
            agent = agent_class(env.action_space)
            return (env, agent)

        self._envs_and_agents = [make_env_and_agent() for _ in range(n_envs)]
        self._network = network_fn()

    def _batch_coroutines(self, cors):
        """Batches a list of coroutines into one.

        Handles waiting for the slowest coroutine and filling blanks in
        prediction requests.
        """
        # Store the final episodes in a list.
        episodes = [None] * len(cors)

        def store_transitions(i, cor):
            episodes[i] = yield from cor
            # End with an infinite stream of Nones, so we don't have
            # to deal with StopIteration later on.
            while True:
                yield None
        cors = [store_transitions(i, cor) for(i, cor) in enumerate(cors)]

        def all_finished(xs):
            return all(x is None for x in xs)

        def batch_requests(xs):
            assert xs

            # Request used as a filler for coroutines that have already
            # finished.
            filler = next(x for x in xs if x is not None)
            # Fill with 0s for easier debugging.
            filler = data.nested_map(np.zeros_like, filler)

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

        requests = [next(cor) for (i, cor) in enumerate(cors)]
        while not all_finished(requests):
            responses = yield batch_requests(requests)
            requests = [
                cor.send(inp)
                for (i, (cor, inp)) in enumerate(
                    zip(cors, unbatch_responses(responses))
                )
            ]
        return episodes

    def run_episode_batch(self, params):
        self._network.params = params
        episode_cor = self._batch_coroutines([
            agent.solve(env)
            for (env, agent) in self._envs_and_agents
        ])
        try:
            inputs = next(episode_cor)
            while True:
                predictions = self._network.predict(inputs)
                inputs = episode_cor.send(predictions)
        except StopIteration as e:
            episodes = e.value
            assert len(episodes) == len(self._envs_and_agents)
            return episodes

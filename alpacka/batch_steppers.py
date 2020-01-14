"""Environment steppers."""

import gin
import numpy as np
import ray

from alpacka import data


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

    def run_episode_batch(self, params, **solve_kwargs):  # pylint: disable=missing-param-doc
        """Runs a batch of episodes using the given network parameters.

        Args:
            params (Network-dependent): Network parameters.
            **solve_kwargs (dict): Keyword arguments passed to Agent.solve().

        Returns:
            List of completed episodes (Agent/Trainer-dependent).
        """
        raise NotImplementedError


class _AgentRequestHandler:
    """Handles requests from the agent coroutine to the network."""

    def __init__(self, network_fn):
        """Initializes AgentRequestHandler.

        Args:
            network_fn (callable): Function () -> Network.
        """
        self.network_fn = network_fn

        self._network = None  # Lazy initialize if needed
        self._new_params_flag = None

    def run_coroutine(self, episode_cor, params):  # pylint: disable=missing-param-doc
        """Runs an episode coroutine using the given network parameters.

        Args:
            episode_cor (coroutine): Agent.solve coroutine.
            params (Network-dependent): Network parameters.

        Returns:
            List of completed episodes (Agent/Trainer-dependent).
        """
        self._new_params_flag = True

        try:
            request = next(episode_cor)
            while True:
                if isinstance(request, data.NetworkRequest):
                    request_handler = self._handle_network_request
                else:
                    request_handler = self._handle_prediction_request
                response = request_handler(request, params)
                request = episode_cor.send(response)
        except StopIteration as e:
            return e.value  # episodes

    def _handle_network_request(self, request, params):
        del request
        return data.NetworkRequest(self.network_fn, params)

    def _handle_prediction_request(self, request, params):
        return self.get_network(params).predict(request)

    def get_network(self, params=None):
        if self._network is None:
            self._network = self.network_fn()
        if params is not None and self._new_params_flag:
            self.network.params = params
            self._new_params_flag = False
        return self._network
    network = property(get_network)


class _NetworkRequestBatcher:
    """Batches network requests."""

    def __init__(self, requests):
        self._requests = requests
        self._model_request = None

    @property
    def batched_request(self):
        """Determines model request and returns it."""
        if self._model_request is not None:
            return self._model_request
        self._model_request = next(x for x in self._requests if x is not None)
        return self._model_request

    def unbatch_responses(self, x):
        return (x if req is not None else None for req in self._requests)


class _PredictionRequestBatcher:
    """Batches prediction requests."""

    def __init__(self, requests):
        self._requests = requests
        self._n_agents = len(requests)
        self._batched_request = None

    @property
    def batched_request(self):
        """Batches requests and returns batched request."""
        if self._batched_request is not None:
            return self._batched_request

        # Request used as a filler for coroutines that have already
        # finished.
        filler = next(x for x in self._requests if x is not None)
        # Fill with 0s for easier debugging.
        filler = data.nested_map(np.zeros_like, filler)

        # Substitute the filler for Nones.
        self._requests = [x if x is not None else filler
                          for x in self._requests]

        def assert_not_scalar(x):
            assert np.array(x).shape, (
                'All arrays in a PredictRequest must be at least rank 1.'
            )
        data.nested_map(assert_not_scalar, self._requests)

        def flatten_first_2_dims(x):
            return np.reshape(x, (-1,) + x.shape[2:])

        # Stack instead of concatenate to ensure that all requests have
        # the same shape.
        self._batched_request = data.nested_stack(self._requests)
        # (n_agents, n_requests, ...) -> (n_agents * n_requests, ...)
        self._batched_request = data.nested_map(flatten_first_2_dims,
                                                self._batched_request)
        return self._batched_request

    def unbatch_responses(self, x):
        def unflatten_first_2_dims(x):
            return np.reshape(
                x, (self._n_agents, -1) + x.shape[1:]
            )
        # (n_agents * n_requests, ...) -> (n_agents, n_requests, ...)
        return data.nested_unstack(
            data.nested_map(unflatten_first_2_dims, x)
        )


@gin.configurable
class LocalBatchStepper(BatchStepper):
    """Batch stepper running locally.

    Runs batched prediction for all Agents at the same time.
    """

    def __init__(self, env_class, agent_class, network_fn, n_envs):
        super().__init__(env_class, agent_class, network_fn, n_envs)

        def make_env_and_agent():
            env = env_class()
            agent = agent_class()
            return (env, agent)

        self._envs_and_agents = [make_env_and_agent() for _ in range(n_envs)]
        self._request_handler = _AgentRequestHandler(network_fn)

    def _get_request_batcher(self, requests):
        """Determines requests common type (all requests must be of the same
        type!) and returns batcher."""
        model_request = next(x for x in requests if x is not None)
        if isinstance(model_request, data.NetworkRequest):
            request_batcher = _NetworkRequestBatcher(requests)
        else:
            request_batcher = _PredictionRequestBatcher(requests)
        return request_batcher

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

        requests = [next(cor) for cor in cors]
        while not all_finished(requests):
            batcher = self._get_request_batcher(requests)
            responses = yield batcher.batched_request
            requests = [
                cor.send(inp)
                for cor, inp in zip(cors, batcher.unbatch_responses(responses))
            ]
        return episodes

    def run_episode_batch(self, params, **solve_kwargs):
        episode_cor = self._batch_coroutines([
            agent.solve(env, **solve_kwargs)
            for (env, agent) in self._envs_and_agents
        ])
        return self._request_handler.run_coroutine(episode_cor, params)


@gin.configurable
class RayBatchStepper(BatchStepper):
    """Batch stepper running remotely using Ray.

    Runs predictions and steps environments for all Agents separately in their
    own workers.

    It's highly recommended to pass params to run_episode_batch as a numpy array
    or a collection of numpy arrays. Then each worker can retrieve params with
    zero-copy operation on each node.
    """

    class Worker:
        """Ray actor used to step agent-environment-network in own process."""

        def __init__(self, env_class, agent_class, network_fn, config):
            # TODO(pj): Test that skip_unknown is required!
            gin.parse_config(config, skip_unknown=True)

            self.env = env_class()
            self.agent = agent_class()
            self.request_handler = _AgentRequestHandler(network_fn)

        def run(self, params, solve_kwargs):
            """Runs the episode using the given network parameters."""
            episode_cor = self.agent.solve(self.env, **solve_kwargs)
            return self.request_handler.run_coroutine(episode_cor, params)

    def __init__(self, env_class, agent_class, network_fn, n_envs):
        super().__init__(env_class, agent_class, network_fn, n_envs)

        config = RayBatchStepper._get_config(env_class, agent_class, network_fn)
        ray_worker_cls = ray.remote(RayBatchStepper.Worker)

        if not ray.is_initialized():
            ray.init()
        self.workers = [ray_worker_cls.remote(  # pylint: disable=no-member
            env_class, agent_class, network_fn, config) for _ in range(n_envs)]

    def run_episode_batch(self, params, **solve_kwargs):
        params_id = ray.put(params, weakref=True)
        solve_kwargs_id = ray.put(solve_kwargs, weakref=True)
        episodes = ray.get([w.run.remote(params_id, solve_kwargs_id)
                            for w in self.workers])
        return episodes

    @staticmethod
    def _get_config(env_class, agent_class, network_fn):
        """Returns gin operative config for (at least) env, agent and network.

        It creates env, agent and network to initialize operative gin-config.
        It deletes them afterwords.
        """

        env_class()
        agent_class()
        network_fn()
        return gin.operative_config_str()

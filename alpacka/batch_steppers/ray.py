"""Environment steppers."""

import typing

import gin
import numpy as np

from alpacka.batch_steppers import core

# WA for: https://github.com/ray-project/ray/issues/5250
# One of later packages (e.g. gym_sokoban.envs) imports numba internally.
# This WA ensures its done before Ray to prevent llvm assertion error.
# TODO(pj): Delete the WA with new Ray release that updates pyarrow.
import numba  # pylint: disable=wrong-import-order
import ray  # pylint: disable=wrong-import-order
del numba


class RayObject(typing.NamedTuple):
    """Keeps value and id of an object in the Ray Object Store."""
    id: typing.Any
    value: typing.Any

    @classmethod
    def from_value(cls, value, weakref=False):
        return cls(ray.put(value, weakref=weakref), value)


class RayBatchStepper(core.BatchStepper):
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
            self._request_handler = core.RequestHandler(network_fn)

        def run(self, params, solve_kwargs):
            """Runs the episode using the given network parameters."""
            episode_cor = self.agent.solve(self.env, **solve_kwargs)
            return self._request_handler.run_coroutine(episode_cor, params)

        @property
        def network(self):
            return self._request_handler.network

    def __init__(self, env_class, agent_class, network_fn, n_envs):
        super().__init__(env_class, agent_class, network_fn, n_envs)

        config = RayBatchStepper._get_config(env_class, agent_class, network_fn)
        ray_worker_cls = ray.remote(RayBatchStepper.Worker)

        if not ray.is_initialized():
            ray.init()
        self.workers = [ray_worker_cls.remote(  # pylint: disable=no-member
            env_class, agent_class, network_fn, config) for _ in range(n_envs)]

        self._params = RayObject(None, None)
        self._solve_kwargs = RayObject(None, None)

    def run_episode_batch(self, params, **solve_kwargs):
        # Optimization, don't send the same parameters again.
        if not np.all(params == self._params.value):
            self._params = RayObject.from_value(params)

        # Optimization, don't send the same solve kwargs again.
        if not solve_kwargs == self._solve_kwargs.value:
            self._solve_kwargs = RayObject.from_value(solve_kwargs)

        episodes = ray.get([w.run.remote(self._params.id, self._solve_kwargs.id)
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

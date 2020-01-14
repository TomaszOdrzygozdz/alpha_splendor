"""Tests for alpacka.batch_steppers."""

import copy
import functools
import platform
import random

from unittest import mock

import gin
import gym
import numpy as np
import pytest

from alpacka import agents
from alpacka import batch_steppers
from alpacka import data
from alpacka import envs
from alpacka import networks

# WA for: https://github.com/ray-project/ray/issues/5250
# One of later packages (e.g. gym_sokoban.envs) imports numba internally.
# This WA ensures its done before Ray to prevent llvm assertion error.
# TODO(pj): Delete the WA with new Ray release that updates pyarrow.
import numba  # pylint: disable=wrong-import-order
import ray  # pylint: disable=wrong-import-order
del numba


class _TestEnv(gym.Env):

    observation_space = gym.spaces.Discrete(1000)
    action_space = gym.spaces.Discrete(1000)

    def __init__(self, actions, n_steps, observations, rewards):
        super().__init__()
        self._actions = actions
        self._n_steps = n_steps
        self._observations = observations
        self._rewards = rewards
        self._step = 0

    def reset(self):
        return self._observations.pop(0)

    def step(self, action):
        self._actions.append(action)
        self._step += 1
        # Assert that we don't do any steps after "done".
        assert self._step <= self._n_steps
        # End the episode at random times.
        done = random.random() < 0.5 or self._step == self._n_steps
        if not done:
            obs = self._observations.pop(0)
        else:
            # Don't take the last observation from the queue, so all sequences
            # are of the same length.
            obs = self.observation_space.sample()
        reward = self._rewards.pop(0)
        return (obs, reward, done, {})


class _TestAgent(agents.OnlineAgent):

    def __init__(
        self,
        observations,
        max_n_requests,
        requests,
        responses,
        actions,
    ):
        super().__init__()
        self._observations = observations
        self._max_n_requests = max_n_requests
        self._requests = requests
        self._responses = responses
        self._actions = actions

    def act(self, observation):
        self._observations.append(observation)
        for _ in range(self._max_n_requests):
            # End the predictions at random times.
            if random.random() < 0.5:
                break
            response = yield np.array([self._requests.pop(0)])
            self._responses.append(response[0])
        return (self._actions.pop(0), {})


class _TestNetwork(networks.DummyNetwork):

    def __init__(self, inputs, outputs):
        tensor_sig = data.TensorSignature(shape=(1,))
        super().__init__(
            network_signature=data.NetworkSignature(
                input=tensor_sig, output=tensor_sig
            )
        )
        self._inputs = inputs
        self._outputs = outputs

    def predict(self, inputs):
        outputs = []
        for x in inputs:
            if x == 0:
                outputs.append(0)
            else:
                self._inputs.append(x)
                outputs.append(self._outputs.pop(0))
        return np.array(outputs)


def _mock_ray_remote(cls):
    class _NewCls:
        def __init__(self, *args, **kwargs):
            self.orig_obj = cls(*args, **kwargs)

        @classmethod
        def remote(cls, *args, **kwargs):
            """Mock Ray Actor factory method."""
            return cls(*args, **kwargs)

        def __getattr__(self, name):
            """Mock every Ray Actor method."""
            orig_attr = self.orig_obj.__getattribute__(name)
            new_attr = mock.Mock()
            new_attr.remote = mock.Mock(side_effect=orig_attr)
            return new_attr

    return _NewCls


def _mock_ray_put_get(x, *args, **kwargs):
    del args
    del kwargs
    return x


def _mock_ray_init(*args, **kwargs):
    del args
    del kwargs


@mock.patch('ray.remote', _mock_ray_remote)
@mock.patch('ray.get', _mock_ray_put_get)
@mock.patch('ray.put', _mock_ray_put_get)
@mock.patch('ray.init', _mock_ray_init)
@pytest.mark.parametrize('batch_stepper_cls', [
    batch_steppers.LocalBatchStepper,
    batch_steppers.RayBatchStepper
])
@pytest.mark.parametrize('max_n_requests', [0, 1, 4])
def test_batch_steppers_run_episode_batch(max_n_requests,
                                          batch_stepper_cls):
    n_envs = 8
    max_n_steps = 4
    n_total_steps = n_envs * max_n_steps
    n_total_requests = n_total_steps * max_n_requests

    # Generate some random data.
    def sample_seq(n):
        return [np.random.randint(1, 999) for _ in range(n)]

    def setup_seq(n):
        expected = sample_seq(n)
        to_return = copy.copy(expected)
        actual = []
        return (expected, to_return, actual)
    (expected_rew, rew_to_return, _) = setup_seq(n_total_steps)
    (expected_obs, obs_to_return, actual_obs) = setup_seq(n_total_steps)
    (expected_act, act_to_return, actual_act) = setup_seq(n_total_steps)
    (expected_req, req_to_return, actual_req) = setup_seq(n_total_requests)
    (expected_res, res_to_return, actual_res) = setup_seq(n_total_requests)

    # Connect all pipes together.
    stepper = batch_stepper_cls(
        env_class=functools.partial(
            _TestEnv,
            actions=actual_act,
            n_steps=max_n_steps,
            observations=obs_to_return,
            rewards=rew_to_return,
        ),
        agent_class=functools.partial(
            _TestAgent,
            observations=actual_obs,
            max_n_requests=max_n_requests,
            requests=req_to_return,
            responses=actual_res,
            actions=act_to_return,
        ),
        network_fn=functools.partial(
            _TestNetwork,
            inputs=actual_req,
            outputs=res_to_return,
        ),
        n_envs=n_envs,
    )
    episodes = stepper.run_episode_batch(params=None)
    transition_batch = data.nested_concatenate(
        # pylint: disable=not-an-iterable
        [episode.transition_batch for episode in episodes]
    )

    # Assert that all data got passed around correctly.
    assert len(actual_obs) >= n_envs
    np.testing.assert_array_equal(actual_obs, expected_obs[:len(actual_obs)])
    np.testing.assert_array_equal(actual_req, expected_req[:len(actual_req)])
    np.testing.assert_array_equal(actual_res, expected_res[:len(actual_req)])
    np.testing.assert_array_equal(actual_act, expected_act[:len(actual_obs)])

    # Assert that we collected the correct transitions (order is mixed up).
    assert set(transition_batch.observation.tolist()) == set(actual_obs)
    assert set(transition_batch.action.tolist()) == set(actual_act)
    assert set(transition_batch.reward.tolist()) == set(
        expected_rew[:len(actual_obs)]
    )
    assert transition_batch.done.sum() == n_envs


@pytest.mark.parametrize('batch_stepper_cls', [
    batch_steppers.LocalBatchStepper,
    batch_steppers.RayBatchStepper
])
def test_batch_steppers_network_request_handling(batch_stepper_cls):
    # Set up
    network_class = networks.DummyNetwork
    network_fn = functools.partial(network_class, network_signature=None)
    xparams = 'params'
    episode = 'yoghurt'
    n_envs = 3

    class TestAgent:
        def solve(self, _):
            network_fn, params = yield data.NetworkRequest()
            assert isinstance(network_fn(), network_class)
            assert params == xparams
            return episode

    # Run
    bs = batch_stepper_cls(
        env_class=envs.CartPole,
        agent_class=TestAgent,
        network_fn=network_fn,
        n_envs=n_envs
    )

    # Test
    episodes = bs.run_episode_batch(xparams)
    assert episodes == [episode] * n_envs


class _TestWorker(batch_steppers.RayBatchStepper.Worker):
    def get_state(self):
        return self.env, self.agent, self.request_handler.network


@mock.patch('alpacka.batch_steppers.RayBatchStepper.Worker', _TestWorker)
@pytest.mark.skipif(platform.system() == 'Darwin',
                    reason='Ray does not work on Mac, see awarelab/alpacka#27')
def test_ray_batch_stepper_worker_members_initialization_with_gin_config():
    # Set up
    solved_at = 7
    env_class = envs.CartPole
    agent_class = agents.RandomAgent
    network_class = networks.DummyNetwork
    n_envs = 3

    gin.bind_parameter('CartPole.solved_at', solved_at)

    env = env_class()
    env.reset()
    root_state = env.clone_state()

    # Run
    bs = batch_steppers.RayBatchStepper(
        env_class=env_class,
        agent_class=agent_class,
        network_fn=functools.partial(network_class, network_signature=None),
        n_envs=n_envs
    )
    bs.run_episode_batch(None,
                         init_state=root_state,
                         time_limit=10)

    # Test
    assert env.solved_at == solved_at
    assert len(bs.workers) == n_envs
    for worker in bs.workers:
        env, agent, network = ray.get(worker.get_state.remote())
        assert isinstance(env, env_class)
        assert isinstance(agent, agent_class)
        assert isinstance(network, network_class)
        assert env.solved_at == solved_at

# TODO(koz4k): Test collecting real/model transitions.

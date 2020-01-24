"""Shooting agent.

It does Monte Carlo simulation."""

import asyncio
import math

import gin
import gym
import numpy as np

from alpacka import batch_steppers
from alpacka import data
from alpacka.agents import base
from alpacka.agents import core


@gin.configurable
def mean_aggregate(act_n, episodes):
    scores = np.zeros(act_n)
    counts = np.zeros(act_n)
    for episode in episodes:
        scores[episode.transition_batch.action[0]] += episode.return_
        counts[episode.transition_batch.action[0]] += 1
    counts[counts == 0] = 1
    return scores / counts


@gin.configurable
def max_aggregate(act_n, episodes):
    scores = np.empty(act_n)
    for i in range(act_n):
        scores[i] = -np.inf
    for episode in episodes:
        action = episode.transition_batch.action[0]
        scores[action] = max(scores[action], episode.return_)
    return scores


class ShootingAgent(base.OnlineAgent):
    """Monte Carlo simulation agent."""

    def __init__(
        self,
        n_rollouts=1000,
        rollout_time_limit=None,
        aggregate_fn=mean_aggregate,
        batch_stepper_class=batch_steppers.LocalBatchStepper,
        agent_class=core.RandomAgent,
        n_envs=10
    ):
        """Initializes ShootingAgent.

        Args:
            n_rollouts (int): Do at least this number of MC rollouts per act().
            rollout_time_limit (int): Maximum number of timesteps for rollouts.
            aggregate_fn (callable): Aggregates simulated episodes. Signature:
                (n_act, episodes) -> np.ndarray(action_scores).
            batch_stepper_class (type): BatchStepper class.
            agent_class (type): Rollout agent class.
            n_envs (int): Number of parallel environments to run.
        """
        super().__init__()
        self._n_rollouts = n_rollouts
        self._rollout_time_limit = rollout_time_limit
        self._aggregate_fn = aggregate_fn
        self._batch_stepper_class = batch_stepper_class
        self._agent_class = agent_class
        self._n_envs = n_envs
        self._model = None
        self._batch_stepper = None
        self._network_fn = None
        self._params = None

    def reset(self, env, observation):
        """Reinitializes the agent for a new environment."""
        assert isinstance(env.action_space, gym.spaces.Discrete), (
            'ShootingAgent only works with Discrete action spaces.'
        )
        yield from super().reset(env, observation)

        self._model = env
        self._network_fn, self._params = yield data.NetworkRequest()

    @asyncio.coroutine
    def act(self, observation):
        """Runs n_rollouts simulations and chooses the best action."""
        assert self._model is not None, (
            'Reset ShootingAgent first.'
        )
        del observation

        # Lazy initialize batch stepper
        if self._batch_stepper is None:
            self._batch_stepper = self._batch_stepper_class(
                env_class=type(self._model),
                agent_class=self._agent_class,
                network_fn=self._network_fn,
                n_envs=self._n_envs,
            )

        root_state = self._model.clone_state()

        # TODO(pj): Move it to BatchStepper. You should be able to query
        # BatchStepper for a given number of episodes (by default n_envs).
        global_n_rollouts = math.ceil(self._n_rollouts / self._n_envs)
        episodes = []
        for _ in range(global_n_rollouts):
            episodes.extend(self._batch_stepper.run_episode_batch(
                params=self._params,
                init_state=root_state,
                time_limit=self._rollout_time_limit,
            ))

        # Aggregate episodes into scores.
        action_scores = self._aggregate_fn(self._action_space.n, episodes)

        # Calculate simulation policy entropy.
        agent_info_batch = data.nested_concatenate(
            [episode.transition_batch.agent_info for episode in episodes])
        if 'entropy' in agent_info_batch:
            sample_entropy = np.mean(agent_info_batch['entropy'])
        else:
            sample_entropy = None

        # Choose greedy action.
        action = np.argmax(action_scores)
        return action, {'sim_pi_entropy': sample_entropy}

    def network_signature(self, observation_space, action_space):
        agent = self._agent_class()
        return agent.network_signature(observation_space, action_space)

    @staticmethod
    def compute_metrics(episodes):
        # Calculate simulation policy entropy.
        agent_info_batch = data.nested_concatenate(
            [episode.transition_batch.agent_info for episode in episodes])
        if np.all(agent_info_batch['sim_pi_entropy']):
            sample_sim_pi_entropy = np.mean(
                agent_info_batch['sim_pi_entropy'])
            sample_sim_pi_entropy_std = np.std(
                agent_info_batch['sim_pi_entropy'])

            return {
                'simulation_entropy': sample_sim_pi_entropy,
                'simulation_entropy_std': sample_sim_pi_entropy_std
            }
        return {}

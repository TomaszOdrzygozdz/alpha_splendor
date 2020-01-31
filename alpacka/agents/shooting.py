"""Shooting agent.

It does Monte Carlo simulation."""

import asyncio
import collections
import math

import gin
import gym
import numpy as np

from alpacka import batch_steppers
from alpacka import data
from alpacka.agents import base
from alpacka.agents import core

# Basic returns aggregators.
gin.external_configurable(np.max, module='np')
gin.external_configurable(np.mean, module='np')


@gin.configurable
@asyncio.coroutine
def truncated_return(episode):
    """Returns sum of rewards up to the truncation of the episode."""
    return episode.return_


@gin.configurable
def bootstrap_return(episode):
    """Bootstraps a state value at the end of the episode if truncated."""
    # TODO(pj): Move this inference to concurrent workers where the agent solves
    # the environment. E.g. "last_value" in data.Episode for ActorCritic?
    return_ = episode.return_
    if episode.truncated:
        batched_value, _ = yield np.expand_dims(
            episode.transition_batch.next_observation[-1], axis=0)
        return_ += batched_value[0, 0]
    return return_


class ShootingAgent(base.OnlineAgent):
    """Monte Carlo simulation agent."""

    def __init__(
        self,
        n_rollouts=1000,
        rollout_time_limit=None,
        aggregate_fn=np.mean,
        estimate_fn=truncated_return,
        batch_stepper_class=batch_steppers.LocalBatchStepper,
        agent_class=core.RandomAgent,
        n_envs=10,
        **kwargs
    ):
        """Initializes ShootingAgent.

        Args:
            n_rollouts (int): Do at least this number of MC rollouts per act().
            rollout_time_limit (int): Maximum number of timesteps for rollouts.
            aggregate_fn (callable): Aggregates simulated episodes returns.
                Signature: list: returns -> float: action score.
            estimate_fn (bool): Simulated episode return estimator.
                Signature: Episode -> float: return. It must be a coroutine.
            batch_stepper_class (type): BatchStepper class.
            agent_class (type): Rollout agent class.
            n_envs (int): Number of parallel environments to run.
            kwargs: OnlineAgent init keyword arguments.
        """
        super().__init__(**kwargs)
        self._n_rollouts = n_rollouts
        self._rollout_time_limit = rollout_time_limit
        self._aggregate_fn = aggregate_fn
        self._estimate_fn = estimate_fn
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
                output_dir=None,
            )

        # TODO(pj): Move it to BatchStepper. You should be able to query
        # BatchStepper for a given number of episodes (by default n_envs).
        episodes = []
        for _ in range(math.ceil(self._n_rollouts / self._n_envs)):
            episodes.extend(self._batch_stepper.run_episode_batch(
                params=self._params,
                epoch=self._epoch,
                init_state=self._model.clone_state(),
                time_limit=self._rollout_time_limit,
            ))

        # Computer episode returns and put them in a map.
        act_to_rets_map = collections.defaultdict(list)
        for episode in episodes:
            return_ = yield from self._estimate_fn(episode)
            act_to_rets_map[episode.transition_batch.action[0]].append(return_)

        # Aggregate episodes into action scores.
        action_scores = np.empty(self._action_space.n)
        for action, returns in act_to_rets_map.items():
            action_scores[action] = (self._aggregate_fn(returns)
                                     if returns else np.nan)

        # Computes action histograms as normalized action scores.
        action_scores = np.nan_to_num(action_scores)
        action_histograms = action_scores / action_scores.sum()

        # Calculate simulation policy entropy.
        agent_info_batch = data.nested_concatenate(
            [episode.transition_batch.agent_info for episode in episodes])
        if 'entropy' in agent_info_batch:
            sample_entropy = np.mean(agent_info_batch['entropy'])
        else:
            sample_entropy = None

        # Calculate the MC estimate of state value.
        value = sum(
            episode.return_ for episode in episodes
        ) / len(episodes)

        agent_info = {
            'action_histogram': action_histograms,
            'sim_pi_entropy': sample_entropy,
            'value': value
        }

        # Choose greedy action.
        action = np.nanargmax(action_scores)

        return action, agent_info

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

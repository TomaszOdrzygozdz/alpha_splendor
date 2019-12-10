"""Monte Carlo Tree Search agent."""

import functools
import math

import gin
import gym
import numpy as np

from alpacka import batch_steppers
from alpacka import networks
from alpacka.agents import base
from alpacka.agents import core


@gin.configurable
def mean_aggregate(act_n, episodes):
    scores = np.zeros(act_n)
    counts = np.zeros(act_n)
    for episode in episodes:
        scores[episode.transition_batch[0].action] += episode.return_
        counts[episode.transition_batch[0].action] += 1
    return scores / counts


@gin.configurable
def max_aggregate(act_n, episodes):
    scores = np.empty(act_n)
    for i in range(act_n):
        scores[i] = -np.inf
    for episode in episodes:
        action = episode.transition_batch[0].action
        scores[action] = max(scores[action], episode.return_)
    return scores


class ShootingAgent(base.OnlineAgent):
    """Monte Carlo simulation agent."""

    def __init__(
        self,
        action_space,
        n_passes=1000,
        aggregate_fn=mean_aggregate,
        batch_stepper_class=batch_steppers.LocalBatchStepper,
        agent_class=core.RandomAgent,
        n_envs=10
    ):
        """Initializes MCTSAgent.

        Args:
            action_space (gym.Space): Action space.
            n_passes (int): Do at least this number of MC simulations per act().
            aggregate_fn (callable): Aggregates simulated episodes. Signature:
                (n_act, episodes) -> np.ndarray(action_scores).
            batch_stepper_class (type): BatchStepper class.
            agent_class (type): Rollout agent class.
            n_envs (int): Number of parallel environments to run.
        """
        assert isinstance(action_space, gym.spaces.Discrete), (
            'ShootingAgent only works with Discrete action spaces.'
        )
        super().__init__(action_space)

        self.n_passes = n_passes
        self.aggregate_fn = aggregate_fn
        self._batch_stepper_class = batch_stepper_class
        self._agent_class = agent_class
        self._n_envs = n_envs
        self._model = None
        self._batch_stepper = None

    def reset(self, env):
        """Reinitializes the agent for a new environment."""
        assert env.action_space == self._action_space
        self._model = env

    def act(self, observation):
        """Runs n_passes simulations and chooses the best action."""
        assert self._model is not None, (
            'Reset ShootingAgent first.'
        )

        # TODO(pj): Request network_fn and params here with yield.
        network_fn = functools.partial(networks.DummyNetwork, input_shape=None)
        params = None

        # Lazy initialize batch stepper
        if self._batch_stepper is None:
            self._batch_stepper = self._batch_stepper_class(
                env_class=type(self._model),
                agent_class=self._agent_class,
                network_fn=network_fn,
                n_envs=self._n_envs,
            )

        root_state = self._model.clone_state()

        # TODO(pj): Move it to BatchStepper. You should be able to query
        # BatchStepper for a given number of episodes (by default n_envs).
        global_n_passes = math.ceil(self.n_passes / self._n_envs)
        episodes = []
        for _ in range(global_n_passes):
            episodes.extend(
                self._batch_stepper.run_episode_batch(params, root_state))

        # Aggregate episodes into scores.
        action_scores = self.aggregate_fn(self._action_space.n, episodes)

        # Choose greedy action.
        action = np.argmax(action_scores)
        return action

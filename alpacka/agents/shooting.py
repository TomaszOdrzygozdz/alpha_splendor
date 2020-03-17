"""Shooting agents.

It does Monte Carlo simulation."""

import asyncio
import math

import gin
import gym
import numpy as np

from alpacka import batch_steppers
from alpacka import data
from alpacka import metric_logging
from alpacka.agents import base
from alpacka.agents import core
from alpacka.utils.transformations import discount_cumsum

# Basic returns aggregators.
gin.external_configurable(np.max, module='np')
gin.external_configurable(np.mean, module='np')


@gin.configurable
@asyncio.coroutine
def truncated_return(episodes, discount=1.):
    """Returns sum of rewards up to the truncation of the episode."""
    return np.array([
        discount_cumsum(episode.transition_batch.reward, discount)[0]
        for episode in episodes
    ])


@gin.configurable
def bootstrap_return_with_value(episodes, discount=1.):
    """Bootstraps a state value at the end of the episode if truncated."""
    last_values, _ = yield np.array([
        episode.transition_batch.next_observation[-1]
        for episode in episodes
    ])
    last_values = np.squeeze(last_values, axis=1)
    returns_ = [
        discount_cumsum(episode.transition_batch.reward, discount)[0]
        for episode in episodes
    ]

    returns_ = [
        return_ + final_value if episode.truncated else return_
        for episode, return_, final_value in
        zip(episodes, returns_, last_values)
    ]
    return returns_


class ShootingAgent(base.OnlineAgent):
    """Monte Carlo prediction agent.

    Uses first-visit Monte Carlo prediction for estimating action qualities
    in the current state.
    """

    def __init__(
        self,
        n_rollouts=1000,
        rollout_time_limit=None,
        aggregate_fn=np.mean,
        estimate_fn=truncated_return,
        batch_stepper_class=batch_steppers.LocalBatchStepper,
        agent_class=core.RandomAgent,
        n_envs=10,
        discount=1.,
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
            discount (float): Future reward discount factor (also known as
                gamma)
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
        self._discount = discount
        self._model = None
        self._batch_stepper = None
        self._network_fn = None
        self._params = None

        self._sim_agent = agent_class()

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

        # Compute episode returns and put them in a map.
        returns_ = yield from self._estimate_fn(episodes, self._discount)
        act_to_rets_map = {key: [] for key in range(self._action_space.n)}
        for episode, return_ in zip(episodes, returns_):
            act_to_rets_map[episode.transition_batch.action[0]].append(return_)

        # Aggregate episodes into action scores.
        action_scores = np.empty(self._action_space.n)
        for action, returns in act_to_rets_map.items():
            action_scores[action] = (self._aggregate_fn(returns)
                                     if returns else np.nan)

        # Calculate the estimate of a state value.
        value = sum(returns_) / len(returns_)

        # Choose greedy action (ignore NaN scores).
        action = np.nanargmax(action_scores)
        onehot_action = np.zeros_like(action_scores)
        onehot_action[action] = 1

        # Pack statistics into agent info.
        agent_info = {
            'action_histogram': onehot_action,
            'value': value,
            'qualities': np.nan_to_num(action_scores),
        }

        # Calculate simulation policy entropy, average value and logits.
        agent_info_batch = data.nested_concatenate(
            [episode.transition_batch.agent_info for episode in episodes])
        if 'entropy' in agent_info_batch:
            agent_info['sim_pi_entropy'] = np.mean(agent_info_batch['entropy'])
        if 'value' in agent_info_batch:
            agent_info['sim_pi_value'] = np.mean(agent_info_batch['value'])
        if 'logits' in agent_info_batch:
            agent_info['sim_pi_logits'] = np.mean(agent_info_batch['logits'])

        self._run_agent_callbacks(episodes)
        return action, agent_info

    def network_signature(self, observation_space, action_space):
        return self._sim_agent.network_signature(
            observation_space, action_space)

    def postprocess_transitions(self, transitions):
        rewards = [transition.reward for transition in transitions]
        discounted_returns = discount_cumsum(rewards, self._discount)

        for transition, discounted_return in zip(
            transitions, discounted_returns
        ):
            transition.agent_info['discounted_return'] = discounted_return

        return transitions

    def _run_agent_callbacks(self, episodes):
        for episode in episodes:
            for callback in self._callbacks:
                callback.on_pass_begin()
                transition_batch = episode.transition_batch
                for ix in range(len(transition_batch.action)):
                    step_agent_info = {
                        key: value[ix]
                        for key, value in transition_batch.agent_info.items()
                    }
                    callback.on_model_step(
                        agent_info=step_agent_info,
                        action=transition_batch.action[ix],
                        observation=transition_batch.next_observation[ix],
                        reward=transition_batch.reward[ix],
                        done=transition_batch.done[ix]
                    )
                callback.on_pass_end()

    @staticmethod
    def compute_metrics(episodes):
        metrics = {}
        agent_info_batch = data.nested_concatenate(
            [episode.transition_batch.agent_info for episode in episodes])

        if 'sim_pi_entropy' in agent_info_batch:
            metrics.update(metric_logging.compute_scalar_statistics(
                agent_info_batch['sim_pi_entropy'],
                prefix='simulation_entropy',
                with_min_and_max=True
            ))

        if 'sim_pi_value' in agent_info_batch:
            metrics.update(metric_logging.compute_scalar_statistics(
                agent_info_batch['sim_pi_value'],
                prefix='network_value',
                with_min_and_max=True
            ))

        if 'sim_pi_logits' in agent_info_batch:
            metrics.update(metric_logging.compute_scalar_statistics(
                agent_info_batch['sim_pi_logits'],
                prefix='network_logits',
                with_min_and_max=True
            ))

        if 'value' in agent_info_batch:
            metrics.update(metric_logging.compute_scalar_statistics(
                agent_info_batch['value'],
                prefix='simulation_value',
                with_min_and_max=True
            ))

        if 'qualities' in agent_info_batch:
            metrics.update(metric_logging.compute_scalar_statistics(
                agent_info_batch['qualities'],
                prefix='simulation_qualities',
                with_min_and_max=True
            ))

        return metrics


class FlatPUCBAgent(ShootingAgent):
    """Flat pUCB agent.

    Uses pUCB formula for estimating action qualities and action probabilities
    from action counts in the current state.
    """

    def __init__(
        self,
        prior_noise=None,
        noise_weight=0.25,
        exploration_weight=1.25,
        **kwargs
    ):
        """Initializes FlatPUCBAgent.

        Args:
            prior_noise (float): Dirichlet distribution parameter alpha.
                Controls how strong is noise added to a prior policy. If None,
                then disabled.
            noise_weight (float): Proportion in which prior noise is added.
            exploration_weight (float): The parameter exploration_weight >= 0
                controls the trade-off between choosing lucrative nodes (low
                weight) and exploring nodes with low visit counts (high weight).
                Rule of thumb: 7 / avg. number of legal moves.
            kwargs: ShootingAgent init keyword arguments.
        """
        super().__init__(**kwargs)
        self._prior_noise = prior_noise
        self._noise_weight = noise_weight
        self._exploration_weight = exploration_weight

    def act(self, observation):
        """Runs n_rollouts simulations and chooses the best action."""
        assert self._model is not None, (
            'Reset FlatPUCBAgent first.'
        )

        # Save the root state
        root_state = self._model.clone_state()

        # Run a prior agent.
        yield from self._sim_agent.reset(self._model, observation)
        _, sim_agent_info = yield from self._sim_agent.act(observation)

        prior_probs = sim_agent_info['prob']
        if self._prior_noise is not None:
            # Add dirichlet noise according to AlphaZero paper.
            prior_probs = (
                (1 - self._noise_weight) * prior_probs +
                self._noise_weight * np.random.dirichlet(
                    [self._prior_noise, ] * len(prior_probs))
            )

        # Lazy initialize batch stepper
        if self._batch_stepper is None:
            self._batch_stepper = self._batch_stepper_class(
                env_class=type(self._model),
                agent_class=self._agent_class,
                network_fn=self._network_fn,
                n_envs=self._n_envs,
                output_dir=None,
            )

        def evaluate(state):
            """Evaluates state, returns an estimated state value."""
            episodes = self._batch_stepper.run_episode_batch(
                params=self._params,
                epoch=self._epoch,
                init_state=state,
                time_limit=self._rollout_time_limit,
            )

            returns_ = yield from self._estimate_fn(episodes, self._discount)
            return np.mean(returns_)

        class Bandit:
            """Bandit stores additional information to moves."""

            def __init__(self, prior_prob):
                self._prior_prob = prior_prob

                self._count = 0
                self._quality = 0

            @property
            def count(self):
                return self._count

            @property
            def quality(self):
                return self._quality

            def visit(self, return_):
                """Updates bandit quality."""

                self._count += 1
                self._quality += (return_ - self._quality) / self._count

            def utility(self, total_count, exploration_weight=1.):
                """Returns bandit UCB1.

                Args:
                    total_count (int): How many times in total any bandit was
                        chosen.
                    exploration_weight (float): The parameter exploration_weight
                        >= 0 controls the trade-off between choosing lucrative
                        nodes (low weight) and exploring nodes with low visit
                        counts (high weight).

                Return:
                    Bandit utility according to pUCB formula.
                """

                return self._quality + \
                    exploration_weight * self._prior_prob * \
                    np.sqrt(total_count) / (1 + self._count)

        # Run flat-UCB for n_rollouts iterations.
        bandits = [Bandit(prior_prob) for prior_prob in prior_probs]
        for i in range(self._n_rollouts):
            # 1. Select bandit.
            action_utilities = [
                bandit.utility(i, exploration_weight=self._exploration_weight)
                for bandit in bandits]
            action = np.argmax(action_utilities)

            # 2. Play bandit.
            _, reward, done, _ = self._model.step(action)
            if done:
                value = 0.
            else:
                value = yield from evaluate(self._model.clone_state())

            # 3. Update bandit.
            return_ = reward + self._discount * value
            bandits[action].visit(return_)

            # Restore model.
            self._model.restore_state(root_state)

        # Calculate the estimate of a state value.
        action_counts, qualities = np.array(
            [[bandit.count, bandit.quality] for bandit in bandits]).transpose()
        value = np.sum(action_counts * qualities) / np.sum(action_counts)

        # Choose greedy action.
        action = np.argmax(action_counts)

        # Pack statistics into agent info.
        agent_info = {
            'action_histogram': action_counts / np.sum(action_counts),
            'value': value,
            'qualities': qualities,
        }

        # Calculate simulation policy entropy, average value and logits.
        if 'entropy' in sim_agent_info:
            agent_info['sim_pi_entropy'] = sim_agent_info['entropy']
        if 'value' in sim_agent_info:
            agent_info['sim_pi_value'] = sim_agent_info['value']
        if 'logits' in sim_agent_info:
            agent_info['sim_pi_logits'] = sim_agent_info['logits']

        self._run_agent_callbacks(episodes)
        return action, agent_info

    @staticmethod
    def compute_metrics(episodes):
        metrics = ShootingAgent.compute_metrics(episodes)
        agent_info_batch = data.nested_concatenate(
            [episode.transition_batch.agent_info for episode in episodes])

        if 'action_histogram' in agent_info_batch:
            search_pi = agent_info_batch['action_histogram']
            search_entropy = -np.nansum(search_pi * np.log(search_pi), axis=-1)
            metrics.update(metric_logging.compute_scalar_statistics(
                search_entropy,
                prefix='search_entropy',
                with_min_and_max=True
            ))

        return metrics

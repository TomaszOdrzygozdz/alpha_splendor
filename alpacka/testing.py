"""Testing utilities."""

import functools

import gym
import numpy as np

from alpacka import data
from alpacka import envs
from alpacka import networks


class TabularEnv(envs.ModelEnv):
    """Tabular environment with hardcoded transitions.

    Observations are equal to states.
    """

    def __init__(self, init_state, n_actions, transitions):
        """Initializes TabularEnv.

        Args:
            init_state (any): Initial state, returned from reset().
            n_actions (int): Number of actions.
            transitions (dict): Dict of structure:
                {
                    state: {
                        action: (state', reward, done),
                        # ...
                    },
                    # ...
                }
        """
        self.observation_space = gym.spaces.Discrete(len(transitions))
        self.action_space = gym.spaces.Discrete(n_actions)
        self._init_state = init_state
        self._transitions = transitions
        self._state = None

    def reset(self):
        self._state = self._init_state
        return self._state

    def step(self, action):
        (self._state, reward, done) = self._transitions[self._state][action]
        return (self._state, reward, done, {})

    def clone_state(self):
        return self._state

    def restore_state(self, state):
        self._state = state


def run_without_suspensions(coroutine):
    try:
        next(coroutine)
        assert False, 'Coroutine should return immediately.'
    except StopIteration as e:
        return e.value


def run_with_constant_network_prediction(coroutine, logits):
    try:
        next(coroutine)
        coroutine.send(logits)
        assert False, 'Coroutine should return after the first prediction.'
    except StopIteration as e:
        return e.value


def run_with_dummy_network_prediction(coroutine, network_signature):
    """Runs a coroutine with a dummy network.

    Args:
        coroutine: Coroutine yielding network requests.
        network_signature (NetworkSignature or None): Signature of the network
            to emulate, or None if the coroutine should not need a network.

    Returns:
        Return value of the coroutine.
    """
    try:
        request = next(coroutine)
        while True:
            batch_size = request.shape[0]
            assert network_signature is not None, 'Coroutine needs a network.'
            output_sig = network_signature.output
            response = zero_pytree(output_sig, shape_prefix=(batch_size,))
            request = coroutine.send(response)
    except StopIteration as e:
        return e.value


def run_with_dummy_network_response(coroutine):
    try:
        next(coroutine)
        coroutine.send((
            functools.partial(networks.DummyNetwork, network_signature=None),
            None
        ))
        assert False, 'Coroutine should return after one request.'
    except StopIteration as e:
        return e.value


def construct_episodes(actions, rewards, **kwargs):
    """Constructs episodes from actions and rewards nested lists.

    Args:
        actions = [
            [a00, a01, a02, ...], # Actions in the first episode.
            [a10, a11, a12, ...], # Actions in the second episode.
            ...
        ]
        rewards = [
            [r00, r01, r02, ...], # Rewards in the first episode.
            [r10, r11, r12, ...], # Rewards in the second episode.
            ...
        ]
        **kwargs (dict): Keyword arguments passed to Episode.

    Note:
        Transition observations and next observations are set to None.
        Done flag is True only for the last transition in the episode.
        Episode.return_ is calculated as an undiscounted sum of rewards.
    """
    episodes = []
    for acts, rews in zip(actions, rewards):
        transitions = [
            # TODO(koz4k): Initialize using kwargs.
            data.Transition(None, act, rew, False, None, {})
            for act, rew in zip(acts[:-1], rews[:-1])]
        transitions.append(
            data.Transition(None, acts[-1], rews[-1], True, None, {}))
        transition_batch = data.nested_stack(transitions)
        episodes.append(data.Episode(transition_batch, sum(rews), **kwargs))
    return episodes


def zero_pytree(signature, shape_prefix=()):
    """Builds a zero-filled pytree of a given signature.

    Args:
        signature (pytree): Pytree of TensorSignature.
        shape_prefix (tuple): Shape to be prepended to each constructed array's
            shape.

    Returns:
        Pytree of a given signature with zero arrays as leaves.
    """
    return data.nested_map(
        lambda sig: np.zeros(shape=shape_prefix + sig.shape, dtype=sig.dtype),
        signature,
    )

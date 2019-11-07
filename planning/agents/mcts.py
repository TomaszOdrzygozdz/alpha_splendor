"""Monte Carlo Tree Search agent."""

import gym

from planning.agents import base
from planning.agents import core


def rate_new_leaves_with_rollouts(
    leaf,
    observation,
    model,
    discount,
    rollout_agent_class=core.RandomAgent,
    rollout_time_limit=100,
):
    agent = rollout_agent_class(model.action_space)
    init_state = model.clone_state()

    child_qualities = []
    for init_action in range(model.action_space.n):
        (observation, reward, done, _) = model.step(init_action)
        quality = reward
        total_discount = 1
        time = 0
        while not done and time < rollout_time_limit:
            action = yield from agent.act(observation)
            (observation, reward, done, _) = model.step(action)
            total_discount *= discount
            quality += total_discount * reward
            time += 1
        child_qualities.append(quality)
        model.restore_state(init_state)
    return child_qualities


class TreeNode:
    """Node of the search tree.

    Attrs:
        quality: Instead of value, so we can handle rating with both V-networks
            and Q-networks. Quality(s, a) = reward(s, a) + discount * value(s').
        reward: Reward estimate accumulated across all visits, so we can handle
            non-deterministic environments.
    """

    def __init__(self, init_quality):
        self._quality_sum = 0
        self._quality_count = 0
        self.children = None
        if init_quality is not None:
            self.add_quality(init_quality)

    def add_quality(self, quality):
        self._quality_sum += quality
        self._quality_count += 1

    @property
    def quality(self):
        return self._quality_sum / self._quality_count

    @property
    def is_leaf(self):
        return self.children is None


class MCTSAgent(base.OnlineAgent):
    """Monte Carlo Tree Search agent."""

    def __init__(
        self,
        n_passes=10,
        discount=0.99,
        rate_new_leaves_fn=rate_new_leaves_with_rollouts,
    ):
        self._n_passes = n_passes
        self._discount = discount
        self._rate_new_leaves = rate_new_leaves_with_rollouts
        self._model = None
        self._root = None
        self._root_state = None

    def _rate_children(self, node):
        return [
            node.children[action].quality
            for action in range(len(node.children))
        ]

    def _choose_action(self, node):
        # TODO(koz4k): Distinguish exploratory/not.
        child_qualities = self._rate_children(node)
        (_, action) = max(zip(child_qualities, range(len(child_qualities))))
        return action

    def _traverse(self, root, observation):
        """Chooses a path from the root to a leaf in the search tree.

        Returns:
            Pair (observation, done, path) where path is a list of pairs
            (reward, node), including leaf, such that reward is obtained when
            transitioning _to_ node. In case of a "done" state, traversal is
            interrupted.
        """
        path = []
        path.append((0, root))
        node = root
        done = False
        while not node.is_leaf and not done:
            action = self._choose_action(node)
            node = node.children[action]
            (observation, reward, done, _) = self._model.step(action)
            path.append((reward, node))
        return (observation, done, path)

    def _expand_leaf(self, leaf, observation):
        """Expands a leaf and returns its quality.

        The leaf's new children are assigned initial qualities, but they're not
        backpropagated yet. They will be only when we expand those new leaves.

        Returns:
            Quality of the expanded leaf.
        """
        assert leaf.is_leaf
        # TODO(koz4k): Check for loops here.
        child_qualities = yield from self._rate_new_leaves(
            leaf, observation, self._model, self._discount
        )
        leaf.children = [TreeNode(quality) for quality in child_qualities]
        return leaf.quality

    def _backpropagate(self, quality, path):
        for (reward, node) in reversed(path):
            quality = reward + self._discount * quality
            node.add_quality(quality)

    def _run_pass(self, root, observation):
        (observation, done, path) = self._traverse(root, observation)
        (reward, leaf) = path[-1]
        if not done:
            quality = yield from self._expand_leaf(leaf, observation)
        else:
            # In a "done" state, cumulative future return is 0, so quality is
            # equal to reward.
            quality = reward
        self._backpropagate(quality, path)
        # Go back to the root state.
        self._model.restore_state(self._root_state)

    def reset(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete), (
            'MCTSAgent only works with Discrete action spaces.'
        )
        self._model = env
        # Initialize root with some quality to avoid division by zero.
        self._root = TreeNode(init_quality=0)

    def act(self, observation):
        assert self._model is not None, (
            'MCTSAgent works only in model-based mode.'
        )
        self._root_state = self._model.clone_state()
        for _ in range(self._n_passes):
            yield from self._run_pass(self._root, observation)
        action = self._choose_action(self._root)
        self._root = self._root.children[action]
        return action

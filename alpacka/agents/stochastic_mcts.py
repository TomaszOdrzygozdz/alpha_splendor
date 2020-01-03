"""Monte Carlo Tree Search for stochastic environments."""

import gin
import numpy as np

from alpacka import data
from alpacka.agents import base
from alpacka.agents import core
from alpacka.utils import space as space_utils


@gin.configurable
def rate_new_leaves_with_rollouts(
    leaf,
    observation,
    model,
    discount,
    rollout_agent_class=core.RandomAgent,
    rollout_time_limit=100,
):
    """Basic rate_new_leaves_fn based on rollouts with an Agent.

    Args:
        leaf (TreeNode): Node whose children are to be rated.
        observation (np.ndarray): Observation received at leaf.
        model (gym.Env): Model environment.
        discount (float): Discount factor.
        rollout_agent_class (type): Agent class to use for rollouts.
        rollout_time_limit (int): Maximum number of timesteps for rollouts.

    Yields:
        Network prediction requests.

    Returns:
        list: List of qualities for all actions played from leaf.
    """
    del leaf
    agent = rollout_agent_class()
    init_state = model.clone_state()

    child_qualities = []
    for init_action in space_utils.space_iter(model.action_space):
        (observation, init_reward, done, _) = model.step(init_action)
        yield from agent.reset(model, observation)
        value = 0
        total_discount = 1
        time = 0
        while not done and time < rollout_time_limit:
            (action, _) = yield from agent.act(observation)
            (observation, reward, done, _) = model.step(action)
            value += total_discount * reward
            total_discount *= discount
            time += 1
        child_qualities.append(init_reward + discount * value)
        model.restore_state(init_state)
    return child_qualities


@gin.configurable
def rate_new_leaves_with_value_network(leaf, observation, model, discount):
    """rate_new_leaves_fn based on a value network (observation -> value)."""
    del leaf
    del observation

    init_state = model.clone_state()

    def step_and_rewind(action):
        (observation, reward, done, _) = model.step(action)
        model.restore_state(init_state)
        return (observation, reward, done)

    (observations, rewards, dones) = data.nested_stack([
        step_and_rewind(action)
        for action in space_utils.space_iter(model.action_space)
    ])
    # Run the network to predict values for children.
    values = yield observations
    # (batch_size, 1) -> (batch_size,)
    values = np.reshape(values, -1)
    # Compute the final qualities, masking out the "done" states.
    return list(rewards + discount * values * (1 - dones))


class TreeNode:
    """Node of the search tree.

    Attrs:
        children (list): List of children, indexed by action.
        is_leaf (bool): Whether the node is a leaf, i.e. has not been expanded
            yet.
    """

    def __init__(self, init_quality=None):
        """Initializes TreeNode.

        Args:
            init_quality (float or None): Quality received from
                the rate_new_leaves_fn for this node, or None if it's the root.
        """
        self._quality_sum = 0
        self._quality_count = 0
        if init_quality is not None:
            self.visit(init_quality)
        self.children = None

    def visit(self, quality):
        """Records a visit in the node during backpropagation.

        Args:
            quality (float): Quality accumulated on the path out of the
                node.
        """
        self._quality_sum += quality
        self._quality_count += 1

    @property
    def quality(self):
        """Returns the quality of going into this node in the search tree.

        We use it instead of value, so we can handle dense rewards.
        Quality(s, a) = reward(s, a) + discount * value(s').
        """
        return self._quality_sum / self._quality_count

    @property
    def is_leaf(self):
        return self.children is None


class StochasticMCTSAgent(base.OnlineAgent):
    """Monte Carlo Tree Search for stochastic environments."""

    def __init__(
        self,
        n_passes=10,
        discount=0.99,
        rate_new_leaves_fn=rate_new_leaves_with_rollouts,
    ):
        """Initializes MCTSAgent.

        Args:
            n_passes (int): Number of MCTS passes per act().
            discount (float): Discount factor.
            rate_new_leaves_fn (callable): Coroutine estimating qualities of new
                leaves. Can ask for predictions using a Network. Should return
                qualities for every child of a given leaf node. Signature:
                (leaf, observation, model, discount) -> [quality].
        """
        super().__init__()
        self.n_passes = n_passes
        self._discount = discount
        self._rate_new_leaves = rate_new_leaves_fn
        self._model = None
        self._root = None
        self._root_state = None

    @staticmethod
    def _rate_children(node):
        """Returns qualities of all children of a given node."""
        return [child.quality for child in node.children]

    def _choose_action(self, node):
        """Chooses the action to take in a given node based on child qualities.

        If avoid_loops is turned on, tries to avoid nodes visited on the path
        from the root.

        Args:
            node (TreeNode): Node to choose an action from.

        Returns:
            Action to take.
        """
        # TODO(koz4k): Distinguish exploratory/not.
        child_qualities = self._rate_children(node)
        child_qualities_and_actions = zip(
            child_qualities, range(len(child_qualities))
        )

        (_, action) = max(child_qualities_and_actions)
        return action

    def _traverse(self, root, observation):
        """Chooses a path from the root to a leaf in the search tree.

        Does not modify the nodes.

        Args:
            root (TreeNode): Root of the search tree.
            observation (np.ndarray): Observation received at root.

        Returns:
            Tuple (path, observation, done), where path is a list of pairs
            (reward, node) of nodes visited during traversal and rewards
            collected when stepping into them, observation is the observation
            received in the leaf, done is the "done" flag received when stepping
            into the leaf. In case of a "done", traversal is interrupted.
        """
        path = [(0, root)]
        node = root
        done = False
        while not node.is_leaf and not done:
            action = self._choose_action(node)
            node = node.children[action]
            (observation, reward, done, _) = self._model.step(action)
            path.append((reward, node))
        return (path, observation, done)

    def _expand_leaf(self, leaf, observation, done):
        """Expands a leaf and returns its quality.

        The leaf's new children are assigned initial quality. The quality of the
        "best" new leaf is then backpropagated.

        Only modifies leaf - adds children with new qualities.

        Args:
            leaf (TreeNode): Leaf to expand.
            observation (np.ndarray): Observation received at leaf.
            done (bool): "Done" flag received at leaf.

        Yields:
            Network prediction requests.

        Returns:
            float: Quality of a chosen child of the expanded leaf.
        """
        assert leaf.is_leaf

        if done:
            # In a "done" state, cumulative future return is 0.
            return 0

        child_qualities = yield from self._rate_new_leaves(
            leaf, observation, self._model, self._discount
        )
        leaf.children = [TreeNode(quality) for quality in child_qualities]
        action = self._choose_action(leaf)
        return leaf.children[action].quality

    def _backpropagate(self, quality, path):
        """Backpropagates quality to the root through path.

        Only modifies the qualities of nodes on the path.

        Args:
            quality (float): Quality collected at the leaf.
            path (list): List of (reward, node) pairs, describing a path from
                the root to a leaf.
        """
        for (reward, node) in reversed(path):
            quality = reward + self._discount * quality
            node.visit(quality)

    def _run_pass(self, root, observation):
        """Runs a pass of MCTS.

        A pass consists of:
            1. Tree traversal to find a leaf.
            2. Expansion of the leaf, adding its successor states to the tree
               and rating them.
            3. Backpropagation of the value of the best child of the old leaf.

        During leaf expansion, new children are rated only using
        the rate_new_leaves_fn - no actual stepping into those states in the
        environment takes place for efficiency, so that rate_new_leaves_fn can
        be implemented by running a neural network that rates all children of
        a given node at the same time. In case of a "done", traversal is
        interrupted, the leaf is not expanded and value 0 is backpropagated.

        Args:
            root (TreeNode): Root node.
            observation (np.ndarray): Observation collected at the root.

        Yields:
            Network prediction requests.
        """
        (path, observation, done) = self._traverse(root, observation)
        (_, leaf) = path[-1]
        quality = yield from self._expand_leaf(leaf, observation, done)
        self._backpropagate(quality, path)
        # Go back to the root state.
        self._model.restore_state(self._root_state)

    def reset(self, env, observation):
        """Reinitializes the search tree for a new environment."""
        yield from super().reset(env, observation)
        self._model = env
        self._root = TreeNode()

    def act(self, observation):
        """Runs n_passes MCTS passes and chooses the best action."""
        assert self._model is not None, (
            'MCTSAgent works only in model-based mode.'
        )
        self._root_state = self._model.clone_state()
        for _ in range(self.n_passes):
            yield from self._run_pass(self._root, observation)

        action = self._choose_action(self._root)
        self._root = self._root.children[action]
        return (action, {})

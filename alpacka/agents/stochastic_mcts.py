"""Monte Carlo Tree Search for stochastic environments."""

import random

import gin
import gym

from alpacka import data
from alpacka.agents import base
from alpacka.agents import core


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
        list: List of pairs (reward, value) for all actions played from leaf.
    """
    del leaf
    agent = rollout_agent_class()
    init_state = model.clone_state()

    child_ratings = []
    for init_action in range(model.action_space.n):
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
        child_ratings.append((init_reward, value))
        model.restore_state(init_state)
    return child_ratings


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
        step_and_rewind(action) for action in range(model.action_space.n)
    ])
    # Run the network to predict values for children.
    values = yield observations
    # Compute the final ratings, masking out "done" states.
    return rewards + discount * values * (1 - dones)


class TreeNode:
    """Node of the search tree.

    Attrs:
        children (list): List of children, indexed by action.
        is_leaf (bool): Whether the node is a leaf, i.e. has not been expanded
            yet.
    """

    def __init__(self, init_reward, init_value=None):
        """Initializes TreeNode.

        Args:
            init_reward (float): Reward collected when stepping into the node
                the first time.
            init_value (float or None): Value received from a rate_new_leaves_fn
                for this node, or None if it's the root.
        """
        self._reward_sum = init_reward
        self._reward_count = 1
        self._value_sum = 0
        self._value_count = 0
        self.children = None
        if init_value is not None:
            self._value_sum += init_value
            self._value_count += 1

    def visit(self, reward, value):
        """Records a visit in the node during backpropagation.

        Args:
            reward (float): Reward collected when stepping into the node.
            value (float or None): Value accumulated on the path out of the
                node, or None if value should not be accumulated.
        """
        self._reward_sum += reward
        self._reward_count += 1
        if value is not None:
            self._value_sum += value
            self._value_count += 1

    def quality(self, discount):
        """Returns the quality of going into this node in the search tree.

        We use it instead of value, so we can handle dense rewards.
        Quality(s, a) = reward(s, a) + discount * value(s').
        """
        return self._reward_sum / self._reward_count + discount * (
            self._value_sum / self._value_count
        )

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
            rate_new_leaves_fn (callable): Coroutine estimating rewards and
                values of new leaves. Can ask for predictions using a Network.
                Should return rewards and values for every child of a given leaf
                node. Signature:
                (leaf, observation, model, discount) -> [(reward, value)].
        """
        super().__init__()
        self.n_passes = n_passes
        self._discount = discount
        self._rate_new_leaves = rate_new_leaves_fn
        self._model = None
        self._root = None
        self._root_state = None

    def _rate_children(self, node):
        """Returns qualities of all children of a given node."""
        return [child.quality(self._discount) for child in node.children]

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

        The leaf's new children are assigned initial rewards and values. The
        reward and value of the "best" new leaf is then backpropagated.

        Only modifies leaf - adds children with new rewards and values.

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

        child_ratings = yield from self._rate_new_leaves(
            leaf, observation, self._model, self._discount
        )
        leaf.children = [
            TreeNode(reward, value) for (reward, value) in child_ratings
        ]
        action = self._choose_action(leaf)
        return leaf.children[action].quality(self._discount)

    def _backpropagate(self, value, path):
        """Backpropagates value to the root through path.

        Only modifies the rewards and values of nodes on the path.

        Args:
            value (float or None): Value collected at the leaf, or None if value
                should not be backpropagated.
            path (list): List of (reward, node) pairs, describing a path from
                the root to a leaf.
        """
        for (reward, node) in reversed(path):
            node.visit(reward, value)
            if value is not None:
                value = reward + self._discount * value

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
        path = []
        (path, observation, done) = self._traverse(root, observation)
        (_, leaf) = path[-1]
        quality = yield from self._expand_leaf(leaf, observation, done)
        self._backpropagate(quality, path)
        # Go back to the root state.
        self._model.restore_state(self._root_state)

    def reset(self, env, observation):
        """Reinitializes the search tree for a new environment."""
        assert isinstance(env.action_space, gym.spaces.Discrete), (
            'MCTSAgent only works with Discrete action spaces.'
        )
        yield from super().reset(env, observation)
        self._model = env
        # Initialize root with some reward to avoid division by zero.
        self._root = TreeNode(init_reward=0)

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

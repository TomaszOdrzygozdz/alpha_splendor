"""Monte Carlo Tree Search agent."""

import random

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

    child_ratings = []
    for init_action in range(model.action_space.n):
        (observation, init_reward, done, _) = model.step(init_action)
        value = 0
        total_discount = 1
        time = 0
        while not done and time < rollout_time_limit:
            action = yield from agent.act(observation)
            (observation, reward, done, _) = model.step(action)
            value += total_discount * reward
            total_discount *= discount
            time += 1
        child_ratings.append((init_reward, value))
        model.restore_state(init_state)
    return child_ratings


class TreeNode:
    """Node of the search tree."""

    def __init__(self, init_reward, init_value=None):
        self._reward_sum = init_reward
        self._reward_count = 1
        self._init_value = init_value
        self._graph_node = None
        self.children = None
        self.is_terminal = False

    def init_graph_node(self, graph_node=None):
        assert self._graph_node is None, 'Graph node initialized twice.'
        if graph_node is None:
            graph_node = GraphNode(self._init_value)
        self._graph_node = graph_node

    @property
    def graph_node(self):
        return self._graph_node

    def visit(self, reward, value):
        self._reward_sum += reward
        self._reward_count += 1
        if not self.is_terminal and value is not None:
            assert self.graph_node is not None, (
                'Graph node must be assigned first.'
            )
            self.graph_node.visit(value)

    def quality(self, discount):
        """Returns the quality of going into this node in the search tree.

        We use it instead of value, so we can handle dense rewards.
        Quality(s, a) = reward(s, a) + discount * value(s').
        """
        return self._reward_sum / self._reward_count + (
            self._graph_node.value
            if self._graph_node is not None else self._init_value
        )

    @property
    def is_leaf(self):
        return self.children is None


class GraphNode:
    """Node of the search graph.

    In the graph mode, corresponds to a state in the MDP. Outside of the graph
    mode, corresponds 1-1 to a TreeNode.
    """

    def __init__(self, init_value):
        self._value_sum = 0
        self._value_count = 0
        if init_value is not None:
            self.visit(init_value)
        # TODO(koz4k): Move children here?

    def visit(self, value):
        self._value_sum += value
        self._value_count += 1

    @property
    def value(self):
        return self._value_sum / self._value_count


class DeadEnd(Exception):
    pass


class MCTSAgent(base.OnlineAgent):
    """Monte Carlo Tree Search agent."""

    def __init__(
        self,
        n_passes=10,
        discount=0.99,
        rate_new_leaves_fn=rate_new_leaves_with_rollouts,
        graph_mode=False,
        avoid_loops=False,
        loop_penalty=0,
    ):
        """Initializes MCTSAgent.

        Args:
            n_passes: (int) Number of MCTS passes per act().
            discount: (float) Discount factor.
            rate_new_leaves: Coroutine estimating rewards and values of new
                leaves. Can ask for predictions using a Network. Should return
                rewards and values for every child of a given leaf node.
                Signature:
                (leaf, observation, model, discount) -> [(reward, value)].
            graph_mode: (bool) Turns on using transposition tables, turning the
                search graph from a tree to a DAG.
            avoid_loops: (bool) Prevents going back to states already visited on
                the path from the root.
            loop_penalty: (float) Value backpropagated from "dead ends" - nodes
                from which it's impossible to reach a node that hasn't already
                been visited.
        """
        if avoid_loops:
            assert graph_mode, 'Loop avoidance only works in graph mode.'

        self.n_passes = n_passes
        self._discount = discount
        self._rate_new_leaves = rate_new_leaves_fn
        self._graph_mode = graph_mode
        self._avoid_loops = avoid_loops
        self._loop_penalty = loop_penalty
        self._model = None
        self._root = None
        self._root_state = None
        self._real_visited = None
        self._state_to_graph_node = {}

    def _rate_children(self, node):
        return [
            node.children[action].quality(self._discount)
            for action in range(len(node.children))
        ]

    def _choose_action(self, node, visited):
        # TODO(koz4k): Distinguish exploratory/not.
        child_qualities = self._rate_children(node)
        child_qualities_and_actions = zip(
            child_qualities, range(len(child_qualities))
        )
        if self._avoid_loops:
            child_graph_nodes = [child.graph_node for child in node.children]
            child_qualities_and_actions = [
                (quality, action)
                for (quality, action) in child_qualities_and_actions
                if child_graph_nodes[action] not in visited
            ]
        if child_qualities_and_actions:
            (_, action) = max(child_qualities_and_actions)
            return action
        else:
            # No unvisited child - dead end.
            raise DeadEnd

    def _traverse(self, root, observation, path):
        """Chooses a path from the root to a leaf in the search tree.

        Returns:
            Pair (observation, done, path) where path is a list of pairs
            (reward, node), including leaf, such that reward is obtained when
            transitioning _to_ node. In case of a "done" state, traversal is
            interrupted.
        """
        path.append((0, root))
        visited = {root.graph_node}
        node = root
        done = False
        visited = set()
        while not node.is_leaf and not done:
            action = self._choose_action(node, visited)
            node = node.children[action]
            (observation, reward, done, _) = self._model.step(action)
            path.append((reward, node))
            visited.add(node.graph_node)
        return (observation, done, visited)

    def _expand_leaf(self, leaf, observation, done, visited):
        """Expands a leaf and returns its quality.

        The leaf's new children are assigned initial rewards and values. The
        reward and value of the "best" new leaf is then backpropagated.

        Returns:
            Quality of a chosen child of the expanded leaf, or None if we
            shouldn't backpropagate quality beause the node has already been
            visited.
        """
        # TODO(koz4k): visited/already_visited - clean up.
        assert leaf.is_leaf
        
        if done:
            leaf.is_terminal = True
            # In a "done" state, cumulative future return is 0.
            return 0

        already_visited = False
        if self._graph_mode:
            state = self._model.clone_state()
            graph_node = self._state_to_graph_node.get(state, None)
            leaf.init_graph_node(graph_node)
            if graph_node is not None:
                already_visited = True
            else:
                self._state_to_graph_node[state] = leaf.graph_node
        else:
            leaf.init_graph_node()

        child_ratings = yield from self._rate_new_leaves(
            leaf, observation, self._model, self._discount
        )
        leaf.children = [
            TreeNode(reward, value) for (reward, value) in child_ratings
        ]
        if already_visited:
            # Node has already been visited, don't backpropagate quality.
            return None
        action = self._choose_action(leaf, visited)
        return leaf.children[action].quality(self._discount)

    def _backpropagate(self, value, path):
        print('path:', path)
        for (reward, node) in reversed(path):
            node.visit(reward, value)
            if value is not None:
                value = reward + self._discount * value

    def _run_pass(self, root, observation):
        path = []
        try:
            (observation, done, visited) = self._traverse(
                root, observation, path
            )
            (_, leaf) = path[-1]
            quality = yield from self._expand_leaf(
                leaf, observation, done, visited
            )
        except DeadEnd:
            quality = self._loop_penalty
        self._backpropagate(quality, path)
        # Go back to the root state.
        self._model.restore_state(self._root_state)

    def reset(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete), (
            'MCTSAgent only works with Discrete action spaces.'
        )
        self._model = env
        # Initialize root with some reward to avoid division by zero.
        self._root = TreeNode(init_reward=0)
        self._real_visited = set()

    def act(self, observation):
        assert self._model is not None, (
            'MCTSAgent works only in model-based mode.'
        )
        self._root_state = self._model.clone_state()
        for _ in range(self.n_passes):
            yield from self._run_pass(self._root, observation)

        # Add the root to visited nodes after MCTS passes to ensure it has
        # a graph node assigned.
        self._real_visited.add(self._root.graph_node)

        try:
            action = self._choose_action(self._root, self._real_visited)
        except DeadEnd:
            action = random.randrange(len(self._root.children))
        self._root = self._root.children[action]
        return action

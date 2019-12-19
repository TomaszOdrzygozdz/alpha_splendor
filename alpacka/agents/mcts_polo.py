import gin
import numpy as np

from alpacka import data
from alpacka.agents import base
from alpacka.agents import core


@gin.configurable
class ScalarValueTraits:

    zero = 0.0

    def __init__(self, dead_end_value=-2.0):
        self.dead_end = dead_end_value


class ValueAccumulator:

    def __init__(self, value):
        # Creates and initializes with typical add
        self.add(value)

    def add(self, value):
        """Adds an abstract value to the accumulator.

        Args:
            value: Abstract value to add.
        """
        raise NotImplementedError

    def get(self):
        """Returns the accumulated abstract value for backpropagation.

        May be non-deterministic.  # TODO(pm): What does it mean?
        """
        raise NotImplementedError

    def index(self):
        """Returns an index for selecting the best node."""
        raise NotImplementedError

    def target(self):
        """Returns a target for value function training."""
        raise NotImplementedError

    def count(self):
        """Returns the number of accumulated values."""
        raise NotImplementedError


@gin.configurable
class ScalarValueAccumulator(ValueAccumulator):

    def __init__(self, value):
        self._sum = 0.0
        self._count = 0
        super().__init__(value)

    def add(self, value):
        self._sum += value
        self._count += 1

    def get(self):
        return self._sum / self._count

    def index(self):
        return self.get()

    def target(self):
        return self.get()

    def count(self):
        return self._count



class GraphNode:
    def __init__(self, value_acc,
                 state=None,
                 terminal=False,
                 solved=False,
                 nedges=4):
        self.value_acc = value_acc
        self.rewards = [None] * nedges
        self.state = state
        self.terminal = terminal
        self.solved = solved

# tree node
class TreeNode:
    def __init__(self, node):
        self.node = node
        self.children = {}  # {valid_action: Node}

    @property
    def rewards(self):
        return self.node.rewards

    @property
    def value_acc(self):
        return self.node.value_acc

    @property
    def state(self):
        return self.node.state

    @state.setter
    def state(self, state):
        self.node.state = state

    def expanded(self):
        return True if self.children else False

    @property
    def terminal(self):
        return self.node.terminal

    @property
    def solved(self):
        return self.node.solved

    @terminal.setter
    def terminal(self, terminal):
        self.node.terminal = terminal


@gin.configurable
class MCTSValue(base.OnlineAgent):

    def __init__(self,
                 action_space,
                 gamma=0.99,
                 n_passes=10,
                 avoid_loops=True,
                 ):
        super().__init__(action_space=action_space)
        self._value_traits = ScalarValueTraits()
        self._gamma = gamma
        self._avoid_loops = avoid_loops
        self._state2node = {}
        self._n_passes = n_passes

    def _children_of_state(self, parent_state):
        old_state = self._model.clone_state()

        self._model.restore_state(parent_state)

        def step_and_rewind(action):
            (observation, reward, done, info) = self._model.step(action)
            state = self._model.clone_state()
            solved = 'solved' in info and info['solved']
            self._model.restore_state(parent_state)
            return (observation, reward, done, solved, state)

        results = zip(*[
            step_and_rewind(action)
            for action in range(self._model.action_space.n)
        ])
        self._model.restore_state(old_state)
        return results

    def run_mcts_pass(self, root):
        # search_path = list of tuples (node, action)
        # leaf does not belong to search_path (important for not double counting its value)
        leaf, search_path = self.tree_traversal(root)
        value = yield from self.expand_leaf(leaf)
        self._backpropagate(search_path, value)

    def tree_traversal(self, root):
        node = root
        seen_states = set()
        search_path = []
        while node.expanded():
            seen_states.add(node.state)
            # INFO: if node Dead End, (new_node, action) = (None, None)
            # INFO: _select_child can SAMPLE an action (to break tie)
            states_to_avoid = seen_states if self._avoid_loops else set()
            new_node, action = self._select_child(node, states_to_avoid)  #
            search_path.append((node, action))
            node = new_node
            if new_node is None:  # new_node is None iff node has no unseen children, i.e. it is Dead End
                break
        # at this point node represents a leaf in the tree (and is None for Dead End).
        # node does not belong to search_path.
        return node, search_path

    def _backpropagate(self, search_path, value):
        # Note that a pair
        # (node, action) can have the following form:
        # (Terminal node, None),
        # (Dead End node, None),
        # (TreeNode, action)
        for node, action in reversed(search_path):
            value = td_backup(node, action, value, self._gamma)  # returns value if action is None
            node.value_acc.add(value)

    def _initialize_graph_node(self, initial_value, state, done, solved):
        value_acc = ScalarValueAccumulator(initial_value)
        new_node = GraphNode(value_acc,
                             state=state,
                             terminal=done,
                             solved=solved,
                             nedges=self._action_space.n)
        self._state2node[state] = new_node  # store newly initialized node in _state2node
        return new_node

    def expand_leaf(self, leaf):
        if leaf is None:  # Dead End
            return self._value_traits.dead_end

        if leaf.terminal:  # Terminal state
            return self._value_traits.zero

        # neighbours are ordered in the order of actions: 0, 1, ..., _model.num_actions
        obs, rewards, dones, solved, states = self._children_of_state(
            leaf.state
        )

        value_batch = yield np.array(obs)

        for idx, action in enumerate(range(self._action_space.n)):
            leaf.rewards[idx] = rewards[idx]
            new_node = self._state2node.get(states[idx], None)
            if new_node is None:
                child_value = value_batch[idx] if not dones[idx] else self._value_traits.zero
                new_node = self._initialize_graph_node(
                    child_value, states[idx], dones[idx], solved=solved[idx]
                )
            leaf.children[action] = TreeNode(new_node)

        return leaf.value_acc.get()

    def _child_index(self, parent, action):
        accumulator = parent.children[action].value_acc
        value = accumulator.index()
        return td_backup(parent, action, value, self._gamma)

    def _rate_children(self, node, states_to_avoid):
        assert self._avoid_loops or len(states_to_avoid) == 0, "Should not happen. There is a bug."
        return [
            (self._child_index(node, action), action)
            for action, child in node.children.items()
            if child.state not in states_to_avoid
        ]

    # here UNLIKE alphazero, we choose final action from the root according to value
    def _select_next_node(self, root):
        # INFO: below line guarantees that we do not perform one-step loop (may be considered slight hack)
        states_to_avoid = {root.state} if self._avoid_loops else set()
        values_and_actions = self._rate_children(root, states_to_avoid)
        if not values_and_actions:
            # when there are no children (e.g. at the bottom states of ChainEnv)
            return None, None
        (_, action) = max(values_and_actions)
        return root.children[action], action

    # Select the child with the highest score
    def _select_child(self, node, states_to_avoid):
        values_and_actions = self._rate_children(node, states_to_avoid)
        if not values_and_actions:
            return None, None
        (max_value, _) = max(values_and_actions)
        argmax = [
            action for value, action in values_and_actions if value == max_value
        ]
        # INFO: here can be sampling
        if len(argmax) > 1:  # PM: This works faster
            action = np.random.choice(argmax)
        else:
            action = argmax[0]
        return node.children[action], action

    def reset(self, env, observation):
        self._model = env
        # 'reset' mcts internal variables: _state2node and _model
        self._state2node = {}
        state = self._model.clone_state()
        (value,) = yield np.array([observation])
        # Initialize root.
        graph_node = self._initialize_graph_node(initial_value=value, state=state, done=False, solved=False)
        self._root = TreeNode(graph_node)

    def act(self, observation):
        # perform MCTS passes. each pass = tree traversal + leaf evaluation + backprop
        for _ in range(self._n_passes):
            yield from self.run_mcts_pass(self._root)
        info = {'node': self._root}
        self._root, action = self._select_next_node(self._root)  # INFO: possible sampling for exploration

        return (action, info)

    @staticmethod
    def postprocess_transition(transition):
        node = transition.agent_info['node']
        value = node.value_acc.target().item()
        return transition._replace(agent_info={'value': value})


def td_backup(node, action, value, gamma):
    if action is None:
        return value
    return node.rewards[action] + gamma * value

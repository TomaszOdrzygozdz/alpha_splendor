"""Debug tracing of agent's decisions."""


import collections

from alpacka.agents import base as agents_base


# Data structures for traces.
# ===========================


# Trace of an agent's planning algorithm.
Trace = collections.namedtuple('Trace', [
    'trajectory',  # Trajectory
    'tree',        # TreeNode
])


Trajectory = collections.namedtuple('Trajectory', [
    'init_state',   # State
    'transitions',  # [Transition]
])


# State of the environment.
State = collections.namedtuple('State', [
    'state_info',   # Env-specific
    'node_id',      # int
    'terminal',     # bool
])


transition_fields = [
    'agent_info',  # dict, Agent-specific, from previous state
    'action',      # Env-specific
    'reward',      # float
    'to_state',    # State
]


# Transition in the real environment.
RealTransition = collections.namedtuple('RealTransition', transition_fields + [
    'passes',      # [[ModelTransiition]], from previous state
])


# Transition in the model.
ModelTransition = collections.namedtuple('ModelTransition', transition_fields)


# Node of the planning tree.
TreeNode = collections.namedtuple('TreeNode', [
    'id',          # int
    'agent_info',  # Agent-specific
    'children',    # dict action -> TreeNode
])


class EnvRenderer:
    """Base class for environment renderers."""

    def render_state(self, state_info):
        """Renders state_info to an image."""
        raise NotImplementedError

    def render_action(self, action):
        """Renders action to a string."""
        raise NotImplementedError


class TraceCallback(agents_base.AgentCallback):
    """Callback for collecting traces."""

    def __init__(self):
        self._env = None
        self._trace = None
        # Trajectory trace.
        self._trajectory = None
        self._passes = None
        self._pass = None
        # Tree trace.
        self._tree_nodes = None
        self._current_root_id = None
        self._current_node_id = None

    @property
    def trace(self):
        return self._trace

    def on_episode_begin(self, env, observation):
        """Called in the beginning of a new episode."""
        del observation
        self._env = env
        state_info = getattr(self._env, 'state_info', None)
        self._trajectory = Trajectory(
            init_state=State(
                state_info=state_info,
                node_id=0,
                terminal=False,
            ),
            transitions=[],
        )
        self._passes = []

        self._tree_nodes = {
            0: TreeNode(
                id=0,
                agent_info=None,
                children={},
            ),
        }
        self._current_root_id = 0

    def on_pass_begin(self):
        """Called in the beginning of every planning pass."""
        self._pass = []
        self._current_node_id = self._current_root_id

    def on_model_step(self, agent_info, action, observation, reward, done):
        """Called after every step in the model."""
        del observation

        self._current_node_id = self._traverse_tree(
            self._current_node_id, agent_info, action
        )

        state_info = getattr(self._env, 'state_info', None)
        self._pass.append(ModelTransition(
            agent_info=agent_info,
            action=action,
            reward=reward,
            to_state=State(
                state_info=state_info,
                node_id=self._current_node_id,
                terminal=done,
            ),
        ))

    def on_pass_end(self):
        """Called in the end of every planning pass."""
        self._passes.append(self._pass)
        self._pass = None

    def on_real_step(self, agent_info, action, observation, reward, done):
        """Called after every step in the real environment."""
        del observation

        self._current_root_id = self._traverse_tree(
            self._current_root_id, agent_info, action
        )
        self._current_node_id = self._current_root_id

        state_info = getattr(self._env, 'state_info', None)
        self._trajectory.transitions.append(RealTransition(
            agent_info=agent_info,
            passes=self._passes,
            action=action,
            reward=reward,
            to_state=State(
                state_info=state_info,
                node_id=self._current_root_id,
                terminal=done,
            ),
        ))
        self._passes = []

    def on_episode_end(self):
        """Called in the end of an episode."""
        self._trace = Trace(
            trajectory=self._trajectory,
            tree=self._build_tree(root_id=0),
        )
        self._env = None
        self._trajectory = None
        self._passes = None
        self._pass = None
        self._tree_nodes = None
        self._current_root_id = None
        self._current_node_id = None
        # TODO(pkozakowski): Dump the trace somewhere.

    @property
    def _next_node_id(self):
        return len(self._tree_nodes)

    def _traverse_tree(self, node_id, agent_info, action):
        node = self._tree_nodes[node_id]._replace(agent_info=agent_info)
        self._tree_nodes[node_id] = node
        if action not in node.children:
            node.children[action] = self._next_node_id
            self._tree_nodes[self._next_node_id] = TreeNode(
                id=self._next_node_id,
                agent_info=None,
                children={},
            )
        return node.children[action]

    def _build_tree(self, root_id):
        root = self._tree_nodes[root_id]
        return root._replace(children={
            action: self._build_tree(id)
            for (action, id) in root.children.items()
        })

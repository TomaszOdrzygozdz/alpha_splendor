"""This code generates interactive HTML file with MCTS Tree Visualized"""

import gin
import numpy as np
import os

from alpacka import agents
from tools.tree_visualizer.tree_visualizer import PREAMBLE_FILE, POSTAMBLE_FILE

from splendor.envs.mechanics.state_as_dict import StateAsDict

@gin.configurable
class MCTSTreeCallback(agents.AgentCallback):
    def __init__(self,
                 output_dir='/home/tomasz/ML_Research/alpha_splendor/own_testing/one_side_splendor/renders',
                 decimals = 3,
                 show_unvisited_nodes=False):
        self._original_root = None
        self.value_decimals = decimals
        self.output_dir = output_dir
        self.show_unvisited_nodes = show_unvisited_nodes
        self.color_of_real_node = '"#7BE141"'
        self.color_of_simulated_node = '"#ADD8E6"'

    def on_real_step(self, agent_info, action, observation, reward, done):
        current_root = agent_info['node']
        if self._original_root is None:
            self._original_root = current_root

        nodes, edges, states = self.parse_tree(current_root)
        self.create_html_file(nodes, edges, states, 'mumin')
        assert False

    def parse_node(self, id, value, count, level,  real_step = False):
        color = self.real_node_color if real_step else self.color_of_simulated_node
        try:
            shortened_value = np.round(value, self.value_decimals)[0]
        except:
            shortened_value = round(value, self.value_decimals)

        label = f'"V: {str(shortened_value)} \\n C: {count}"'
        print(f'label = {label}')
        return '{' + f'id : {id}, level : {level}, color: {color}, label : {label}' + '}'

    def parse_edge(self, from_node, to_node, label):
        return '{' + f'from: {from_node}, to: {to_node}, label: "{label}"' + ', font: { align: "middle" }}'

    def parse_tree(self, root):
        nodes_id = {}
        nodes_lvl = {root : 0}
        parent = {root: None}
        parent_action = {root: None}
        parsed_nodes = []
        parsed_edges = []
        parsed_states = []
        queue = [root]
        idx = 0
        while len(queue) > 0:
            node_to_eval = queue.pop(0)
            nodes_id[node_to_eval] = idx
            if node_to_eval.value_acc.count() > 0 or self.show_unvisited_nodes == False:
                for action in node_to_eval.children:
                    child = node_to_eval.children[action]
                    queue.append(child)
                    parent[child] = node_to_eval
                    parent_action[child] = action


            if parent[node_to_eval] is not None:
                nodes_lvl[node_to_eval] = nodes_lvl[parent[node_to_eval]] + 1
                parsed_edges.append(self.parse_edge(nodes_id[node_to_eval], nodes_id[parent[node_to_eval]],
                                                    parent_action[node_to_eval].short_description()))

            parsed_nodes.append(self.parse_node(nodes_id[node_to_eval],
                                                node_to_eval.value_acc.get(),
                                                node_to_eval.value_acc.count(),
                                                nodes_lvl[node_to_eval]))
            parsed_states.append(f'"{StateAsDict(node_to_eval.node.state).__repr__()}"')
            idx += 1

        return parsed_nodes, parsed_edges, parsed_states

    def create_html_file(self, parsed_nodes, parsed_edges, parsed_states, file_name):
        with open(PREAMBLE_FILE, 'r') as file:
            preamble = file.read()
        with open(POSTAMBLE_FILE, 'r') as file:
            postamble = file.read()
        def nodes_to_str():
            return '\n nodes = [' + ','.join(parsed_nodes) + '] \n'
        def edges_to_str():
            return '\n edges = [' + ','.join(parsed_edges) + '] \n'
        def states_to_str():
            return '\n states = [' + ','.join(parsed_states) + '] \n'
        combined = preamble + nodes_to_str() + edges_to_str() + states_to_str() +  postamble + '</body></html>'
        text_file = open(os.path.join(self.output_dir, file_name + '.html'), "w")
        text_file.write(combined)
        text_file.close()
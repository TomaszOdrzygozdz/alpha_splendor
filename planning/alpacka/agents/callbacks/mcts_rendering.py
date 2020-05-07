"""This code generates interactive HTML file with MCTS Tree Visualized"""

import gin
import numpy as np
import os

from alpacka import agents
from tools.tree_visualizer.tree_visualizer import PREAMBLE_FILE, POSTAMBLE_FILE, \
    SUMMARY_PREAMBLE_FILE, SUMMARY_POSTAMBLE_FILE

from splendor.envs.mechanics.board import Board
from splendor.envs.mechanics.players_hand import PlayersHand
from splendor.envs.mechanics.state import State
from splendor.envs.mechanics.state_as_dict import StateAsDict

@gin.configurable
class MCTSTreeCallback(agents.AgentCallback):
    def __init__(self,
                 output_dir='/home/tomasz/ML_Research/alpha_splendor/own_testing/one_side_splendor/renders_new',
                 decimals = 3,
                 show_unvisited_nodes = False):
        self._original_root = None
        self.value_decimals = decimals
        self.level_separation = 3
        self.output_dir = output_dir
        self.show_unvisited_nodes = show_unvisited_nodes
        self.color_of_real_node = '"#7BE141"'
        self.color_of_simulated_node = '"#ADD8E6"'
        self._links = {}
        self._step_number = 0
        self._epoch = 0
        self._real_vertices = set()
        self.renderer = SplendorToHtmlRenderer()

    def on_episode_begin(self, env, observation, epoch):
        self._epoch = epoch

    def on_real_step(self, agent_info, action, observation, reward, done):
        current_root = agent_info['node']
        self._real_vertices.add(current_root)
        self._real_vertices.add(current_root.children[action])
        if self._original_root is None:
            self._original_root = current_root

        nodes, edges, states = self.parse_tree(current_root)
        file_name = f'episode_{self._epoch}/step_{self._step_number}'
        self.create_one_step_html(nodes, edges, states, file_name)
        self._links[self._step_number] = file_name
        self._step_number += 1
        self.create_summary_html(f'episode_{self._epoch}')

    def on_episode_end(self):
        nodes, edges, states = self.parse_tree(self._original_root)
        self.create_one_step_html(nodes, edges, states, f'episode_{self._epoch}/all')
        self._original_root = None
        self._epoch += 1
        self._links = {}
        self._step_number = 0
        self._epoch = 0
        self._real_vertices = set()

    def parse_node(self, id, value, count, level,  real_step = False):
        color = self.color_of_real_node if real_step else self.color_of_simulated_node
        try:
            shortened_value = np.round(value, self.value_decimals)[0]
        except:
            shortened_value = round(value, self.value_decimals)

        label = f'"V: {str(shortened_value)} \\n C: {count}"'
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
            if node_to_eval.value_acc.count() > 1 or self.show_unvisited_nodes == True:
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
                                                self.level_separation*nodes_lvl[node_to_eval],
                                                node_to_eval in self._real_vertices))
            parsed_states.append(f'"{self.renderer.render_state(node_to_eval.node.state)}"')
            idx += 1

        return parsed_nodes, parsed_edges, parsed_states


    def create_one_step_html(self, parsed_nodes, parsed_edges, parsed_states, file_name):
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
        os.makedirs(os.path.join(self.output_dir, f'episode_{self._epoch}'), exist_ok=True)
        with open(os.path.join(self.output_dir, file_name + '.html'), "w") as html_file:
            html_file.write(combined)

    def create_summary_html(self, file_name):
        with open(SUMMARY_PREAMBLE_FILE, 'r') as file:
            preamble = file.read()
        with open(SUMMARY_POSTAMBLE_FILE, 'r') as file:
            postamble = file.read()

        def parse_links():
            links_as_str = f'<a href="episode_{self._epoch}/all.html" target="mcts_tree_window">  ALL   </a>'
            for idx in self._links:
                links_as_str += f'\n <a href="{self._links[idx]}.html" target="mcts_tree_window"> | {idx}  </a> \n'
            links_as_str += '<br>'
            links_as_str += f'<iframe src="episode_{self._epoch}/step_0.html" height="2000" width="1800" name="mcts_tree_window" ' \
                            'id="mcts_tree_window"></iframe>'
            return links_as_str
        combined = preamble + parse_links() +  postamble + '</body></html>'
        text_file = open(os.path.join(self.output_dir, file_name + '.html'), "w")
        text_file.write(combined)
        text_file.close()


class SplendorToHtmlRenderer:
    def board_to_html(self, board : Board):
        board_html = '<font size=5> Board: </font><br>'
        board_html += f'<br><font size=4>Gems on board: {board.gems_on_board} </font> <br>'
        board_html += '<br> <font size=4> Cards on board: </font> <br>'
        for card in board.cards_on_board:
            board_html += f'{card} <br>'
        board_html += '<br> <font size=4> Nobles on board: </font> <br>'
        for noble in board.nobles_on_board:
            board_html += f'{noble} <br>'
        return board_html

    def additional_state_info(self, state : State):
        info_html = '<br><hr><br>'
        info_html += '<font size=5> Additional info: </font><br>'
        info_html += f'is_done = {state.is_done} <br>'
        info_html += f'winner = {state.winner} <br>'
        info_html += f'info = {state.info} <br>'
        info_html += f'step_taken_so_far = {state.steps_taken_so_far} <br> <hr>'
        return info_html

    def player_summary(self, state: State):
        def player_summary(info: str, player: PlayersHand, color : str):
            header_html = '<br>'
            header_html += f'<font color={color} size=5> {info}: <b>{player.name} </b> </font> <br>'
            header_html += f'<font color={color} size=4><b> Points: {player.number_of_my_points()} </b></font>'
            header_html += f'<font color={color} size=4> Cards: {len(player.cards_possessed)} </font>'
            header_html += f'<font color={color} size=4> Reserved: {len(player.cards_reserved)} </font>'
            header_html += f'<font color={color} size=4> Nobles: {len(player.nobles_possessed)} </font> <br>'
            header_html += f'<font color={color} size=4> Gems: <b> {player.gems_possessed} </b> </font> <br>'
            header_html += f'<font color={color} size=4> Discount: {player.discount()} </font> <br>'
            if len(player.cards_reserved)>0:
                for card in player.cards_reserved:
                    if player.can_afford_card(card):
                        header_html += f'<font color={color} size=4> <b>+ {card} </b> </font><br>'
                    else:
                        header_html += f'<font color={color} size=4> <b>- {card} </b> </font> <br>'
            header_html += '<hr>'
            return header_html

        return player_summary('Active: ', state.active_players_hand(), 'red') + \
               player_summary('Other: ', state.other_players_hand(), 'black')


    def render_state(self, state):
        return self.player_summary(state) + self.board_to_html(state.board) + self.additional_state_info(state)

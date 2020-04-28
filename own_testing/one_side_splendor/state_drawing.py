import time

from splendor.envs.graphics.splendor_gui import SplendorGUI
from splendor.envs.mechanics.state_as_dict import StateAsDict

vis = SplendorGUI()
state_as_dict = StateAsDict()
state_as_dict.load_from_dict({'active_player_hand': {'noble_possessed_ids': {106}, 'cards_possessed_ids': {0, 3, 38, 39, 40, 74, 77, 79, 18, 20, 22, 29}, 'cards_reserved_ids': set(), 'gems_possessed': [0, 0, 3, 1, 1, 3], 'name': 'Player A'}, 'other_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': {1, 36, 21}, 'cards_reserved_ids': set(), 'gems_possessed': [0, 3, 1, 3, 3, 0], 'name': 'Player B'}, 'board': {'nobles_on_board': {100, 101}, 'cards_on_board': {4, 5, 68, 37, 6, 11, 14, 17, 83, 51, 85, 62}, 'gems_on_board': [5, 1, 0, 0, 0, 1], 'deck_order': [{'Row.CHEAP': [43, 54, 2, 59, 61, 76, 75, 72, 78, 57, 56, 55, 24, 19, 60, 58, 23, 73, 7, 42, 41, 25]}, {'Row.MEDIUM': [47, 45, 8, 13, 80, 81, 46, 48, 26, 12, 63, 27, 31, 67, 10, 82, 49, 44, 30, 65, 9, 66, 28, 84, 64]}, {'Row.EXPENSIVE': [87, 69, 34, 88, 35, 52, 86, 32, 16, 70, 15, 89, 33, 71, 53, 50]}]}, 'active_player_id': 0})

state_to_draw = state_as_dict.to_state()
vis.draw_state(state_to_draw)
time.sleep(10000)
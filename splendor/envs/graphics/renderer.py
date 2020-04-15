from PIL import Image, ImageDraw, ImageFont
import numpy as np
from splendor.envs.mechanics.players_hand import PlayersHand
from splendor.envs.mechanics.enums import *
from splendor.envs.mechanics.state import State


class SplendorBasicRenderer:
    def __init__(self):
        IMAGE_WIDTH = 1000
        IMAGE_HEIGHT = 800
        BOARD_TITLE_X = 600
        BOARD_TITLE_Y = 20
        PLAYERS_TITLE_X = 300
        PLATERS_TITLE_Y = 20

        self.img = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), color='white')
        self.large_font = ImageFont.truetype("arial.ttf", 25)
        self.medium_font = ImageFont.truetype("arial.ttf", 15)

        self.labels = ImageDraw.Draw(self.img)
        self.labels.text((BOARD_TITLE_X, BOARD_TITLE_Y), "Board", font=self.large_font, fill='black')
        self.labels.text((PLAYERS_TITLE_X, PLATERS_TITLE_Y), "Players", font=self.large_font, fill='black')

         #img.save('pil_text.png')
        st = State()
        self.draw_players_hand(st.active_players_hand(), 100, 100)
        self.show()

    def draw_players_hand(self, players_hand: PlayersHand, x: int, y: int):
        profits_to_show = ''.join([str(color).replace('GemColor.', '') + ':  ' +
                                    str(players_hand.gems_possessed.gems_dict[color]) + '\n' for color in GemColor])
        self.labels.text((x, y), f"Profits:\n{profits_to_show}", font=self.medium_font, fill='black')


    def show(self):
        self.img.show()

SplendorBasicRenderer()
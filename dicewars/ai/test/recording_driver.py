from copy import deepcopy
from os.path import isfile

import numpy as np

import dicewars.ai.test.xlogin00
from dicewars.client.ai_driver import AIDriver, EndTurnCommand

AI_CLASS = dicewars.ai.test.xlogin00.AI

class RecordingDriver(AIDriver):
    BOARD_FILE = "train_data/board.npy"

    def __init__(self, game, ai_constructor, config):
        self.board_state = []
        self.our_ai = -1
        super().__init__(game, ai_constructor, config)

    def process_command(self, command):
        if isinstance(command, EndTurnCommand):
            if self.our_ai == -1:
                if isinstance(self.ai, AI_CLASS):
                    self.our_ai = self.current_player_name

            self.board_state.append(deepcopy(self.board))
        return super().process_command(command)

    def handle_server_message(self, msg):
        if msg['type'] == 'game_end':
            if self.our_ai != -1:
                self.logger.debug("Saving training data")
                self.export_data(msg["winner"])
        return super().handle_server_message(msg)

    def export_data(self, winner):
        boards = []
        if isfile(self.BOARD_FILE):
            boards = list(np.load(self.BOARD_FILE, allow_pickle=True))

        boards.append({
            "our_ai": self.our_ai,
            "board": self.board_state,
            "winner": winner
        })
        np.save(self.BOARD_FILE, boards)

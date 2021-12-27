from copy import deepcopy

from dicewars.server.area import Area
from dicewars.server.game import Game
import numpy as np


class ServerRecord(Game):
    def __init__(self, board, area_ownership, players, game_config, addr, port, nicknames_order):
        self.battle_record = [[]]
        self.transfer_record = [[]]
        self.board_state = []

        super().__init__(board, area_ownership, players, game_config, addr, port, nicknames_order)

    def export_data(self):
        np.save("train_data/battles.npy", self.battle_record)
        np.save("train_data/transfers.npy", self.transfer_record)
        np.save("train_data/board.npy", self.board_state)

    def handle_player_turn(self):
        """
        Handle clients message and carry out the action
        """
        self.logger.debug(
            "Handling player {} ({}) turn".format(self.current_player.get_name(), self.current_player.nickname))
        player = self.current_player.get_name()
        msg = self.get_message(player)

        if msg['type'] == 'battle':
            self.nb_consecutive_end_of_turns = 0

            # copy board before battle
            board_before = deepcopy(self.board)
            battle = self.battle(self.board.get_area_by_name(msg['atk']), self.board.get_area_by_name(msg['def']))

            # copy board after battle
            board_after = deepcopy(self.board)

            self.summary.add_battle()
            self.logger.debug("Battle result: {}".format(battle))

            # append battle to list of battles this turn
            self.battle_record[-1].append(
                {
                    "board_before": board_before,
                    "battle": battle,
                    "board_after": board_after
                }
            )

            for p in self.players:
                self.send_message(self.players[p], 'battle', battle=battle)

        elif msg['type'] == 'end_turn':
            self.nb_consecutive_end_of_turns += 1
            affected_areas = self.end_turn()
            # append array for new turn
            self.battle_record.append([])
            self.transfer_record.append([])
            self.board_state.append({"player": player, "board": self.board})

            for p in self.players:
                self.send_message(self.players[p], 'end_turn', areas=affected_areas)

        elif msg['type'] == 'transfer':
            self.nb_consecutive_end_of_turns = 0
            transfer = self.transfer(self.board.get_area_by_name(msg['src']), self.board.get_area_by_name(msg['dst']))

            # apend transfer info to transfer record array
            self.transfer_record[-1].append(transfer)

            for p in self.players:
                self.send_message(self.players[p], 'transfer', transfer=transfer)

        else:
            self.logger.warning(f'Unexpected message type: {msg["type"]}')

    def process_win(self, player_nick, player_name):
        # Add final board state
        self.board_state.append({"player": self.current_player.get_name(), "board": self.board})
        self.export_data()
        super().process_win(player_nick, player_name)



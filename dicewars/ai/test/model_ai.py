import logging
import random
from os.path import isfile

import numpy as np

from dicewars.ai.test.recording_server import ServerRecord
from dicewars.client.game.board import Board
from dicewars.client.game.area import Area
from typing import Iterator, Tuple
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack, probability_of_holding_area
import torch
from train_data.model_training import MODEL_PATH
from dicewars.ai.kb.move_selection import get_transfer_from_endangered
from dicewars.ai.kb.move_selection import get_transfer_to_border


class AI:
    PLAYER_ARRAY_FILE = "train_data/game_player.npy"

    def __init__(self, player_name, board, players_order, max_transfers):
        self.max_transfers = max_transfers
        self.player_name = player_name
        self.logger = logging.getLogger('NN_AI')
        self.logger.info(player_name)
        self.model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        self.model.eval()
        self.append_name()
        self.ATTACK_THRESHOLD = 0.4
        self.HOLD_THRESHOLD = 0.3
        #self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.DEVICE = "cpu"

    def ai_turn(self,
                board: Board,
                nb_moves_this_turn: int,
                nb_transfers_this_turn: int,
                nb_turns_this_game: int,
                time_left: float
                ):

        if nb_transfers_this_turn + 2 < self.max_transfers:
            transfer = (board, self.player_name)
            if transfer:
                return TransferCommand(transfer[0], transfer[1])
        else:
            self.logger.debug(f'Already did {nb_transfers_this_turn}/{self.max_transfers} transfers, reserving 2 for evac, skipping further aggresive ones')

        attacks = list(possible_attacks(board, self.player_name))
        for attack, defend in attacks:
            features = torch.tensor(self.get_features(board, attack, defend), dtype=torch.float).to(self.DEVICE)
            attack_success, area_hold = self.model(features)
            # print("Attack success: ", attack_success)
            mine = attack.get_dice()
            his = defend.get_dice()
            #if attack_success > self.ATTACK_THRESHOLD: # and area_hold > self.HOLD_THRESHOLD:  # and area_hold > self.HOLD_THRESHOLD:
            if (mine > his) or (mine == 8):
                self.logger.debug(f'Attacking becasue {mine}  >  {his}')
                return BattleCommand(attack.name, defend.name)

        if nb_transfers_this_turn < self.max_transfers:
            transfer = get_transfer_from_endangered(board, self.player_name)
            if transfer:
                return TransferCommand(transfer[0], transfer[1])
        else:
            self.logger.debug(f'Already did {nb_transfers_this_turn}/{self.max_transfers} transfers, skipping further')

        self.logger.debug(f'Ending turn attacks: {attacks}')        
        return EndTurnCommand()

    def get_features(self, board: Board, attack: Area, defend: Area):
        atk_dice = attack.get_dice()
        deff_dice = defend.get_dice()
        atk_neighbours = len([area for area in [board.get_area(name) for name in attack.get_adjacent_areas_names()] if
                              area.get_owner_name() == attack.get_name()])
        atk_neighbour_dice = sum(
            [area.get_dice() for area in [board.get_area(name) for name in attack.get_adjacent_areas_names()] if
             area.get_owner_name() == attack.get_owner_name()]) ## fuj area.dice -> area.get_dice() ?
        deff_neighbours = len(
            [area for area in [board.get_area(name) for name in defend.get_adjacent_areas_names()] if
             area.get_owner_name() == defend.get_owner_name()]) 
        deff_neighbour_dice = sum(
            [area.get_dice() for area in [board.get_area(name) for name in defend.get_adjacent_areas_names()] if
             area.get_owner_name() == defend.get_owner_name()])

        atk_probability = probability_of_successful_attack(board, attack.get_name(), defend.get_name())
        hold_probability = probability_of_holding_area(board, attack.get_name(), attack.get_dice() - 1, attack.get_owner_name())
        deff_dice_normalized = deff_dice / 8
        atk_dice_normalized = atk_dice / 8
        deff_neighbour_dice_normalized = 0 if deff_neighbours == 0 else deff_neighbour_dice / (deff_neighbours * 8)
        atk_neighbour_dice_normalized = 0 if atk_neighbours == 0 else atk_neighbour_dice / (atk_neighbours * 8)

        atk_target_after_attack_hold_prob = probability_of_holding_area(board, defend.get_name(), attack.get_dice() - 1,
                                                                        attack.get_owner_name())
        atk_source_after_attack_hold_prob = probability_of_holding_area(board, defend.get_name(), 1,
                                                                        attack.get_owner_name())

        return (atk_probability,
                hold_probability,
                deff_dice_normalized,
                atk_dice_normalized,
                deff_neighbour_dice_normalized,
                deff_neighbours,
                atk_neighbour_dice_normalized,
                atk_neighbours,
                atk_target_after_attack_hold_prob,
                atk_source_after_attack_hold_prob,
                )

    def append_name(self):
        game_player = []
        if isfile(AI.PLAYER_ARRAY_FILE):
            game_player = list(np.load(AI.PLAYER_ARRAY_FILE))
        game_player.append(self.player_name)
        np.save(AI.PLAYER_ARRAY_FILE, game_player)

import logging

from dicewars.client.game.board import Board
from dicewars.client.game.area import Area
from typing import Iterator, Tuple
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack, probability_of_holding_area


class AI:
    def __init__(self, player_name, board, players_order, max_transfers):
        self.HOLD_PROB_THRESHOLD = 0.1
        self.ATTACK_PROB_THRESHOLD = 0.4
        self.AREA_VULNERABLE_THRESHOLD = 0.6
        self.AREA_OK_THRESHOLD = 1
        self.AREA_OK_TRANSFER_THRESHOLD = 4

        self.max_transfers = max_transfers
        self.player_name = player_name
        self.logger = logging.getLogger('SuperUltraCleverAI')
        self.logger.error(player_name)

        self.stage = "pre-attack"

    def ai_turn(self,
                board: Board,
                nb_moves_this_turn: int,
                nb_transfers_this_turn: int,
                nb_turns_this_game: int,
                time_left: float
                ):

        self.logger.debug(self.stage)

        if self.stage == "pre-attack":
            command = self.pre_attack(board, nb_transfers_this_turn)
            if command is None:
                command = self.attack(board)
                if command is None:
                    return self.post_attack(board, nb_transfers_this_turn)
                return command
            return command

        elif self.stage == "attack":
            command = self.attack(board)
            if command is None:
                return self.post_attack(board, nb_transfers_this_turn)
            self.logger.debug(f"Attacking from: {command.source_name} to: {command.target_name}")
            return command

        elif self.stage == "post-attack":
            return self.post_attack(board, nb_transfers_this_turn)

    def pre_attack(self, board, nb_transfers_this_turn):
        """
        Pre attack phase. Moves dice to vulnerable areas
        """
        command = self.get_border_reinforcement_command(board)

        if nb_transfers_this_turn < self.max_transfers / 2 and command is not None:
            return command
        else:
            self.stage = "attack"

    def attack(self, board):
        """
        Attack phase. Performs attack according to constants
        """
        attacks = self.get_attacks(board)
        if len(attacks) != 0:
            (source, target, prob) = attacks[0]
            return BattleCommand(source.name, target.name)
        else:
            self.stage = "post-attack"

    def post_attack(self, board, nb_transfers_this_turn):
        """
        Post attack phase. Moves dice to vulnerable areas
        """
        command = self.get_border_reinforcement_command(board)
        if nb_transfers_this_turn < self.max_transfers and command is not None:
            return command
        else:
            self.stage = "pre-attack"
            return EndTurnCommand()

    def get_border_reinforcement_command(self, board: Board):
        """
        Gets command for bored reinforcement
        """
        paths = self.get_ok_to_vulnerable_areas_paths(board, self.player_name)
        if paths is None or len(paths) == 0:
            return None

        path = paths[0]
        return TransferCommand(path[0].name, path[1].name)

        # areas = board.get_player_areas(self.player_name)
        # border = board.get_player_border(self.player_name)
        # border_names = [b.get_name() for b in border]
        #
        # commands = []
        #
        # for area in areas:
        #     border_neighbours = set(area.get_adjacent_areas_names()) & set(border_names)
        #     while len(border_neighbours) != 0:
        #         commands.append(TransferCommand(area.name, border_neighbours.pop()))
        #
        # return commands

    def get_attacks(self, board: Board):
        """
        Gets viable attacks
        """

        attacks = possible_attacks(board, self.player_name)
        viable_attacks = []
        for source, target in attacks:
            if probability_of_successful_attack(board, source.name, target.name) > self.ATTACK_PROB_THRESHOLD:
                holding_prob = probability_of_holding_area(board, target.name, source.dice - 1, self.player_name)
                if holding_prob > self.HOLD_PROB_THRESHOLD:
                    viable_attacks.append((source, target, holding_prob))

        return viable_attacks

    def get_path_to_area(self, board: Board, area_from: Area, area_to: Area, searched=None) -> list[Area]:
        """
        Recursive function for getting path to specified area
        """
        if searched is None:
            searched = []

        paths = []

        adjacent = area_from.get_adjacent_areas_names()
        for x in adjacent:
            area_x = board.get_area(x)
            if x in searched or area_x.owner_name != self.player_name:
                continue
            if x == area_to.name:
                return [area_from, area_to]
            else:
                searched.append(x)
                found_path = self.get_path_to_area(board, area_x, area_to, searched)

                if found_path is None or len(found_path) == 0:
                    continue
                else:
                    found_path.insert(0, area_from)
                    paths.append(found_path)

        if len(paths) == 0:
            return None
        else:
            return list(sorted(paths, key=len))[0]

    def get_ok_to_vulnerable_areas_paths(self, board: Board, player: int):
        """
        Get all paths from areas with hold probability of 1 to areas with holding probability lower than the constant
        """
        (vulnerable, ok) = self.get_vulnerable_areas(board, player)
        ok = list(filter(lambda area: area.dice > self.AREA_OK_TRANSFER_THRESHOLD, ok))
        if len(ok) == 0:
            return []

        paths = []
        for area_ok in ok:
            for area_vulnerable in vulnerable:
                path = self.get_path_to_area(board, area_ok, area_vulnerable)
                if path is None:
                    continue
                paths.append(path)

        paths.sort(key=len)
        return paths


    def get_vulnerable_areas(self, board: Board, player: int) -> (list[Area], list[Area]):
        """
        Gets areas with holding probability lower than  value specified in constant
        """
        player_area = board.get_player_areas(player)
        vulnerable = []
        area_ok = []
        for area in player_area:
            prob = probability_of_holding_area(board, area.name, area.dice, player)
            if prob < self.AREA_VULNERABLE_THRESHOLD:
                vulnerable.append(area)
            elif prob == self.AREA_OK_THRESHOLD:
                area_ok.append(area)

        return vulnerable, area_ok

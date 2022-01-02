import logging

import torch

from dicewars.ai.xgalba03.utils import extract_features_from_board, fast_board_copy
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack, attack_succcess_probability
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.client.game.board import Board

MAX_DICE = 8


class AI:
    """Naive player agent

    This agent performs all possible moves in random order
    """

    def __init__(self, player_name, board, players_order, max_transfers):
        """
        Parameters
        ----------
        game : Game
        """
        self.player_name = player_name
        self.players_order = players_order
        self.logger = logging.getLogger('AI')
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.load("dicewars/ai/xgalba03/model.pt", map_location=torch.device(self.DEVICE))
        self.model.eval()
        self.max_transfers = max_transfers

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        """AI agent's turn

        Get a random area. If it has a possible move, the agent will do it.
        If there are no more moves, the agent ends its turn.
        """

        transfer = None
        if nb_transfers_this_turn + 2 < self.max_transfers:
            """my_borders = [a.name for a in board.get_player_border(self.player_name)]
            #my_border_names = [a.name for a in my_borders]
            my_areas = board.get_player_areas(self.player_name)
            #possible_transfers = [a for a in my_areas if a.name not in my_borders and a.get_dice() > 1]
            possible_transfers = []
            second_row = []
            for border_area in my_borders:
                for neigh in board.get_area(border_area).get_adjacent_areas_names():
                    if neigh not in my_borders and board.get_area(neigh) in my_areas and neigh not in possible_transfers:
                        if board.get_area(neigh).get_dice() > 1 and board.get_area(neigh).get_dice() < MAX_DICE:
                            possible_transfers.append(neigh)
                        else:
                            second_row.append(neigh)
                        self.logger.debug(f'---------------------------------------Possible')

            for area in possible_transfers:
                for neigh in board.get_area(area).get_adjacent_areas_names():
                    if neigh in my_borders and board.get_area(neigh).get_dice() < MAX_DICE:
                        transfer = area, neigh
                        self.logger.debug(f'---------------------------------------Transfering')

            #transfer = get_transfer_to_border(board, self.player_name)
            if transfer:
                return TransferCommand(transfer[0], transfer[1])
"""
            my_borders = [a.name for a in board.get_player_border(self.player_name)]
            my_areas = board.get_player_areas(self.player_name)
            second_row = []
            for border_area in my_borders:
                for neigh in board.get_area(border_area).get_adjacent_areas_names():
                    if neigh not in second_row:
                        second_row.append(board.get_area(neigh))
            possible_transfers = [a for a in second_row if
                                  a.name not in my_borders and a in my_areas and a.get_dice() > 1]
            # second_row = []

            for area in possible_transfers:
                for neigh in area.get_adjacent_areas_names():
                    if neigh in my_borders and board.get_area(neigh).get_dice() < MAX_DICE:
                        transfer = area.get_name(), neigh
                        # if neigh not in second_row:
                        #    second_row.append(neigh)

            if transfer:
                return TransferCommand(transfer[0], transfer[1])
            ##still have move left but no transfers available (maybe borders are full)
            # new_borders = [a for board.get_area(a).get_adjacent_areas_names in my_borders]
            possible_transfers = []
            for area in second_row:
                for neigh in area.get_adjacent_areas_names():
                    if neigh not in second_row and board.get_area(
                            neigh) in my_areas and neigh not in possible_transfers:
                        if board.get_area(neigh).get_dice() > 1 and board.get_area(neigh).get_dice() < MAX_DICE:
                            possible_transfers.append(neigh)
                            self.logger.debug(f'---------------------------------------Possible second row')

            for area in possible_transfers:
                for neigh in board.get_area(area).get_adjacent_areas_names():
                    if neigh in my_borders and board.get_area(neigh).get_dice() < MAX_DICE:
                        transfer = area, neigh
                        self.logger.debug(f'---------------------------------------Transfering second row')

            if transfer:
                return TransferCommand(transfer[0], transfer[1])
            #    for neigh in board.get_area(area).get_adjacent_areas_names()
        else:
            self.logger.debug(
                f'Already did {nb_transfers_this_turn}/{self.max_transfers} transfers, reserving 2 for evac, skipping further aggresive ones')

        if time_left < 3:
            self.logger.warning("TIME IS RUNNING OUT !!! using cheaper strategy")
            return self.best_attack(board)

        attacks = list(possible_attacks(board, self.player_name))
        guteAttacks = []
        guteAttackHeuristics = []
        guteAttacksTuples = []
        summs = []
        if attacks:
            for attacker, defender in attacks:
                if (probability_of_successful_attack(board, attacker.get_name(),
                                                     defender.get_name()) >= 0.5 or attacker.get_dice() == defender.get_dice()) and attacker.get_dice() != 1:
                    guteAttacks.append((attacker, defender))
                    guteAttacksTuples.append(
                        self.mixmanM(fast_board_copy(board), attacker.get_name(), defender.get_name(), 0))
                else:
                    pass
                    # self.logger.debug("no nice atk")

        if guteAttacks:
            for attack in guteAttacksTuples:
                summ = 2 * attack[self.players_order.index(self.player_name)] - sum(list(attack))
                summs.append(summ)
            min_summ = min(summs)
            for attack, summ in zip(guteAttacksTuples, summs):
                # summ = attack[self.players_order.index(self.player_name)] - sum(list(attack))
                guteAttackHeuristics.append(
                    probability_of_successful_attack(board, attacker.get_name(), defender.get_name()) * (
                                summ + min_summ + 1))
            attacker, defender = guteAttacks[guteAttackHeuristics.index(max(guteAttackHeuristics))]
            return BattleCommand(attacker.get_name(), defender.get_name())

        if nb_transfers_this_turn < self.max_transfers:
            my_borders = [a.name for a in board.get_player_border(self.player_name)]
            my_areas_names = [a.name for a in board.get_player_areas(self.player_name)]
            transfer = None
            retreats = []

            for area in my_borders:
                area = board.get_area(area)
                my_dice = area.get_dice()
                if my_dice > 1:
                    for neigh in area.get_adjacent_areas_names():
                        if neigh not in my_areas_names:
                            continue
                        neigh_area = board.get_area(neigh)
                        areas = [area, neigh_area]

                        hold_probab_list = []
                        for area in areas:
                            p = 1.0
                            area_dice = area.get_dice()
                            for attacker in area.get_adjacent_areas_names():
                                attacker_area = board.get_area(attacker)
                                att_dice = attacker_area.get_dice()
                                p *= area_dice / (area_dice + att_dice)
                            hold_probab_list.append(p)

                        expected_loss_no_evac = sum((1 - p) * a.get_dice() for p, a in zip(hold_probab_list, areas))

                        src_dice = area.get_dice()
                        dst_dice = neigh_area.get_dice()

                        dice_moved = min(8 - dst_dice, src_dice - 1)
                        area.dice -= dice_moved
                        neigh_area.dice += dice_moved

                        hold_probab_list = []
                        for area in areas:
                            p = 1.0
                            area_dice = area.get_dice()
                            for attacker in area.get_adjacent_areas_names():
                                attacker_area = board.get_area(attacker)
                                att_dice = attacker_area.get_dice()
                                p *= area_dice / (area_dice + att_dice)
                            hold_probab_list.append(p)
                        expected_loss_evac = sum((1 - p) * a.get_dice() for p, a in zip(hold_probab_list, areas))

                        area.set_dice(src_dice)
                        neigh_area.set_dice(dst_dice)

                        retreats.append(((area, neigh_area), expected_loss_no_evac - expected_loss_evac))

            retreats = sorted(retreats, key=lambda x: x[1], reverse=True)

            if retreats:
                retreat = retreats[0]
                if retreat[1] > 0.0:
                    transfer = retreat[0][0].get_name(), retreat[0][1].get_name()

            if transfer:
                return TransferCommand(transfer[0], transfer[1])
        # self.logger.debug("No more gute ataks.")
        return EndTurnCommand()

    def mixmanM(self, board, attackerName, defenderName, zarazkaBro):
        attacker = board.get_area(attackerName)
        defender = board.get_area(defenderName)

        attackingPlayerName = attacker.get_owner_name()

        indexActual = self.players_order.index(attackingPlayerName)
        indexNext = 0 if indexActual + 1 >= len(self.players_order) else indexActual + 1
        nextPlayer = self.players_order[indexNext]

        # zmena boardy za predpokladu ze utok vyjde
        defender.set_owner(attackerName)
        defender.set_dice(attacker.get_dice() - 1)
        attacker.set_dice(1)

        # sumOfAllDice = 0
        # for player in self.players_order:
        #     sumOfAllDice += board.get_player_dice(player)
        # return (probOfAttack * board.get_player_dice(player_name))/sumOfAllDice

        ## spocita se vyhodnost pro danyho sulina

        if zarazkaBro == 1:
            vyslednejTupl = ()
            for player in self.players_order:
                features = extract_features_from_board(board, player, self.players_order)
                features = torch.tensor(features, dtype=torch.float).to(self.DEVICE)
                vyslednejTupl += (float(self.model(features)),)
            return vyslednejTupl
        else:
            zarazkaBro = 1 if nextPlayer == self.player_name else 0

            attacks = list(possible_attacks(board, nextPlayer))

            guteAttacks = []
            guteAttackHeuristics = []
            guteAttacksTuples = []
            bruh = ()
            for player in self.players_order:
                bruh += (0,)
            if attacks:
                for attacker, defender in attacks:
                    if (probability_of_successful_attack(board, attacker.get_name(),
                                                         defender.get_name()) >= 0.5 or attacker.get_dice() == defender.get_dice()) and attacker.get_dice() != 1:
                        guteAttacks.append((attacker, defender))
                        # self.logger.debug(
                        #     "mixmanM: " + str(attacker.get_name()) + " " + str(defender.get_name()) + " " + str(
                        #         zarazkaBro))
                        guteAttacksTuples.append(
                            self.mixmanM(fast_board_copy(board), attacker.get_name(), defender.get_name(), zarazkaBro))
                if guteAttacks:
                    for attack in guteAttacksTuples:
                        # self.logger.debug("uhmm: " + str(attack))
                        guteAttackHeuristics.append(
                            probability_of_successful_attack(board, attacker.get_name(), defender.get_name()) * (
                                    2 * attack[self.players_order.index(attackingPlayerName)] - sum(list(attack))))
                    return guteAttacksTuples[guteAttackHeuristics.index(max(guteAttackHeuristics))]
                else:
                    return bruh
            else:
                return bruh

    def best_attack(self, board: Board):
        attacks = possible_attacks(board, self.player_name)
        good_attacks = []
        for attack, defend in attacks:
            probability = attack_succcess_probability(attack.get_dice(), defend.get_dice())
            if probability > 0.5 or (attack.dice == 8 and defend.dice == 8):
                good_attacks.append((attack, defend, probability))

        good_attacks.sort(key=lambda x: x[2], reverse=True)

        if len(good_attacks) == 0:
            return EndTurnCommand()

        (attack, defend, probability) = good_attacks[0]
        return BattleCommand(attack.get_name(), defend.get_name())

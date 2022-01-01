import logging
import copy

import torch

from dicewars.ai.test.utils import extract_features_from_board
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand


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
        self.model = torch.load("train_data/model.pt", map_location=torch.device(self.DEVICE))
        self.model.eval()

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        """AI agent's turn

        Get a random area. If it has a possible move, the agent will do it.
        If there are no more moves, the agent ends its turn.
        """
        self.logger.info(time_left)

        attacks = list(possible_attacks(board, self.player_name))
        guteAttacks = []
        guteAttackHeuristics = []
        guteAttacksTuples = []
        if attacks:
            for attacker, defender in attacks:
                if (probability_of_successful_attack(board, attacker.get_name(),
                                                     defender.get_name()) >= 0.5 or attacker.get_dice() == defender.get_dice()) and attacker.get_dice() != 1:
                    guteAttacks.append((attacker, defender))
                    guteAttacksTuples.append(
                        self.mixmanM(copy.deepcopy(board), attacker.get_name(), defender.get_name(), 0))
                else:
                    pass
                    # self.logger.debug("no nice atk")

        if guteAttacks:
            for attack in guteAttacksTuples:
                guteAttackHeuristics.append(
                    probability_of_successful_attack(board, attacker.get_name(), defender.get_name()) * (
                            2 * attack[self.players_order.index(self.player_name)] - sum(list(attack))))
            attacker, defender = guteAttacks[guteAttackHeuristics.index(max(guteAttackHeuristics))]
            return BattleCommand(attacker.get_name(), defender.get_name())

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
                            self.mixmanM(copy.deepcopy(board), attacker.get_name(), defender.get_name(), zarazkaBro))
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

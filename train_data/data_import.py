import numpy as np

import dicewars
import dicewars.ai.utils as utils
from dicewars.ai.test.recording_server import ServerRecord
from dicewars.server.area import Area

from dicewars.server.board import Board


def extract_features_from_battle(battle, next_turn_board: Board):
    board_before: Board = battle["board_before"]
    board_after: Board = battle["board_after"]
    battle_stat = battle["battle"]

    atk = battle_stat["atk"]["name"]
    deff = battle_stat["def"]["name"]

    atk_area: Area = board_before.get_area_by_name(atk)
    deff_area: Area = board_before.get_area_by_name(deff)

    deff_area_after = board_after.get_area_by_name(deff)
    atk_area_after = board_after.get_area_by_name(atk)

    atk_dice = atk_area.dice
    deff_dice = deff_area.dice
    atk_neighbours = len([area for area in atk_area.get_adjacent_areas() if area.owner_name == atk_area.owner_name])
    atk_neighbour_dice = sum(
        [area.dice for area in atk_area.get_adjacent_areas() if area.owner_name == atk_area.owner_name])
    deff_neighbours = len([area for area in deff_area.get_adjacent_areas() if area.owner_name != atk_area.owner_name])
    deff_neighbour_dice = sum(
        [area.dice for area in deff_area.get_adjacent_areas() if area.owner_name != atk_area.owner_name])

    atk_probability = probability_of_successful_attack_server(board_before, atk, deff)
    hold_probability = probability_of_holding_area_server(board_before, atk, atk_area.dice - 1, atk_area.owner_name)
    deff_dice_normalized = deff_dice / 8
    atk_dice_normalized = atk_dice / 8
    deff_neighbour_dice_normalized = 0 if deff_neighbours == 0 else deff_neighbour_dice / (deff_neighbours * 8)
    atk_neighbour_dice_normalized = 0 if atk_neighbours == 0 else atk_neighbour_dice / (atk_neighbours * 8)

    atk_target_after_attack_hold_prob = probability_of_holding_area_server(board_before, deff, atk_area.dice - 1,
                                                                           atk_area.owner_name)
    atk_source_after_attack_hold_prob = probability_of_holding_area_server(board_before, deff, 1, atk_area.owner_name)

    atk_win = battle_stat["atk"]["pwr"] > battle_stat["def"]["pwr"]
    atk_held = next_turn_board.get_area_by_name(atk).owner_name == atk_area.owner_name
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
            atk_win,
            atk_held)


def probability_of_successful_attack_server(board: Board, atk_area, target_area):
    """Calculate probability of attack success

    Parameters
    ----------
    board : Board
    atk_area : int
    target_area : int

    Returns
    -------
    float
        Calculated probability
    """
    atk = board.get_area_by_name(atk_area)
    target = board.get_area_by_name(target_area)
    atk_power = atk.get_dice()
    def_power = target.get_dice()
    return utils.attack_succcess_probability(atk_power, def_power)


def probability_of_holding_area_server(board: Board, area_name, area_dice, player_name):
    """Estimate probability of holding an area until next turn

    Parameters
    ----------
    board : Board
    area_name : int
    area_dice : int
    player_name : int
        Owner of the area

    Returns
    -------
    float
        Estimated probability
    """
    area = board.get_area_by_name(area_name)
    probability = 1.0
    for adj in area.get_adjacent_areas_names():
        adjacent_area = board.get_area_by_name(adj)
        if adjacent_area.get_owner_name() != player_name:
            enemy_dice = adjacent_area.get_dice()
            if enemy_dice == 1:
                continue
            lose_prob = utils.attack_succcess_probability(enemy_dice, area_dice)
            hold_prob = 1.0 - lose_prob
            probability *= hold_prob
    return probability


def get_battle_features_for_player_from_game(game, player_filter):
    features = []

    battles = game["battles"]
    board = game["board"]
    for i in range(len(battles)):
        player = board[i]["player"]
        if player != player_filter:
            continue
        y = 1
        while i + y < len(board) and player != board[i + y]["player"]:
            y += 1

        next_turn_board = board[i + y - 1]
        turn = battles[i]
        for battle in turn:
            features.append(extract_features_from_battle(battle, next_turn_board["board"]))
    return features


def get_player_features_from_games():
    features = []
    games = list(np.load(ServerRecord.GAMES_FILE, allow_pickle=True))
    players = list(np.load("train_data/game_player.npy"))
    for game, player in zip(games, players):
        features.append(get_battle_features_for_player_from_game(game, player))

    return features


if __name__ == '__main__':
    features = get_player_features_from_games()
    print(features)

from dicewars.client.game.board import Board

BOARD_SIZE = 35
MAX_DICE = 8


def extract_features_from_board(board: Board, player: int, player_list: list[int]):
    border = board.get_player_border(player)
    border_count = len(border) / BOARD_SIZE
    border_dice = 0 if (len(border) * MAX_DICE) == 0 else sum([area.get_dice() for area in border]) / (
            len(border) * MAX_DICE)

    player_area = board.get_player_areas(player)
    player_area_count = len(player_area) / BOARD_SIZE
    player_dice = 0 if (len(player_area) * MAX_DICE) == 0 else board.get_player_dice(player) / (
            len(player_area) * MAX_DICE)
    player_largest_area_count = max([len(region) for region in board.get_players_regions(player)])

    enemy_area = get_enemy_area(board, player, player_list)
    enemy_dice = 0 if len(enemy_area) == 0 else sum([area.get_dice() for area in enemy_area]) / len(enemy_area)
    enemy_area_count = 0 if BOARD_SIZE == 0 else len(enemy_area) / BOARD_SIZE

    border_enemy_side = get_player_border_enemy_side(board, player, player_list)
    border_enemy_side_count = len(border_enemy_side)
    border_enemy_dice = 0 if (len(border_enemy_side) * MAX_DICE) == 0 else sum(
        [area.get_dice() for area in border_enemy_side]) / (len(border_enemy_side) * MAX_DICE)

    neighbouring_players = get_neighbouring_players(board, player)
    # normalized neighbour count -1 is because current player cannot be neighbour to itself
    neighbours_count = 0 if (len(player_list) - 1) == 0 else len(neighbouring_players) / (len(player_list) - 1)

    return (
        border_count,
        border_dice,
        player_area_count,
        player_largest_area_count,
        player_dice,
        enemy_dice,
        enemy_area_count,
        border_enemy_side_count,
        border_enemy_dice,
        neighbours_count
    )


def get_player_border_enemy_side(board: Board, current_player: int, player_list: list[int]):
    neighbour_areas = []

    for player in player_list:
        if player == current_player:
            continue

        areas = board.get_player_areas(player)
        for area in areas:
            for area_name in area.get_adjacent_areas_names():
                neighbour = board.get_area(area_name)
                if neighbour.get_owner_name() == current_player:
                    neighbour_areas.append(neighbour)

    return neighbour_areas


def get_enemy_area(board: Board, current_player: int, player_list: list[int]):
    area = []
    for player in player_list:
        if player == current_player:
            continue
        area += board.get_player_areas(player)

    return area


def get_neighbouring_players(board: Board, player: int):
    border = board.get_player_border(player)
    neighbouring_players = []
    for area in border:
        for neighbour_name in area.get_adjacent_areas_names():
            neighbour_area = board.get_area(neighbour_name)
            if neighbour_area.get_owner_name() not in neighbouring_players:
                neighbouring_players.append(neighbour_area.get_owner_name())

    return neighbouring_players

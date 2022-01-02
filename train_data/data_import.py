import numpy as np

from train_data.recording_driver import RecordingDriver
from train_data.recording_server import ServerRecord
from dicewars.ai.xgalba03.utils import extract_features_from_board

BOARD_SIZE = 35
MAX_DICE = 8


def get_features_from_games():
    server_games = list(np.load(ServerRecord.GAMES_FILE, allow_pickle=True))
    client_games = list(np.load(RecordingDriver.BOARD_FILE, allow_pickle=True))

    games = []
    for s_game, c_game in zip(server_games, client_games):
        game = {
            "won": won(c_game),
            "board_states": []
        }
        for board in c_game["board"]:
            game["board_states"].append(extract_features_from_board(board, c_game["our_ai"], s_game["players"]))

        games.append(game)

    return games


def won(client_game) -> bool:
    return client_game["our_ai"] == client_game["winner"]


if __name__ == '__main__':
    features = get_features_from_games()
    print(features)
    pass

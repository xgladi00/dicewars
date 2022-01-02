import os
from os.path import isfile

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import train_data.data_import as td
from collections import OrderedDict

from torch.optim import SGD

from dicewars.ai.test.recording_driver import RecordingDriver
from dicewars.ai.test.recording_server import ServerRecord

MODEL_PATH = "train_data/model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


WIN_RATE_WINDOW_SIZE = 20
game_results = [0] * WIN_RATE_WINDOW_SIZE
win_rate_rolling_window = []


def get_model(input_layer_size: int, hidden_layer_size: int, output_layer_size: int):
    model = nn.Sequential(
        OrderedDict([
            ("input", nn.Linear(input_layer_size, hidden_layer_size)),
            ("activation", nn.ReLU()),
            ("output_layer", nn.Linear(hidden_layer_size, output_layer_size)),
        ])
    )

    if isfile(MODEL_PATH):
        print("Loading existing model from: {}".format(MODEL_PATH))
        model = torch.load(MODEL_PATH, map_location=torch.device(DEVICE))
    else:
        print("New model created")
        torch.save(model, MODEL_PATH)

    return model


def add_game_result(result, game_results):
    game_results.append(result)
    game_results = game_results[-WIN_RATE_WINDOW_SIZE:]
    win_rate_rolling_window.append(sum(game_results) / len(game_results))


# DEVICE = "cpu"

def train():
    print("training with: {}".format(DEVICE))
    model = get_model(10, 256, 1).to(DEVICE)

    battle_features = td.get_features_from_games()
    model.train()

    opt = torch.optim.RMSprop(model.parameters())
    loss_func = nn.MSELoss()

    for game in battle_features:
        target_features = torch.tensor([game["won"]], dtype=torch.float).to(DEVICE)
        # add win rate from last 10 games to chart and show it
        add_game_result(game["won"], game_results)
        plt.plot(win_rate_rolling_window)
        plt.show()

        loss_game = []
        for features in game["board_states"]:
            input_features = torch.tensor(features, dtype=torch.float).to(DEVICE)
            opt.zero_grad()

            out = model(input_features)
            loss = loss_func(out, target_features)
            loss_game.append(loss)
            print("predict: {} target: {} loss: {}".format(out, target_features, loss))
            loss.backward()
            opt.step()
        if len(loss_game) != 0:
            print("average error: {}".format(sum(loss_game) / len(loss_game)))

    plt.plot()
    if isfile(ServerRecord.GAMES_FILE):
        os.remove(ServerRecord.GAMES_FILE)
    if isfile(RecordingDriver.BOARD_FILE):
        os.remove(RecordingDriver.BOARD_FILE)

    torch.save(model, MODEL_PATH)


if __name__ == "__main__":
    train()

import os
from os.path import isfile

import numpy as np
import torch.nn as nn
import torch
import train_data.data_import as td
from collections import OrderedDict

from torch.optim import SGD

from dicewars.ai.test.recording_server import ServerRecord


def get_model(input_layer_size: int, hidden_layer_size: int, output_layer_size: int):
    model = nn.Sequential(
        OrderedDict([
            ("hidden_layer", nn.Linear(input_layer_size, hidden_layer_size)),
            ("activation", nn.ReLU()),
            ("output_layer", nn.Linear(hidden_layer_size, output_layer_size))
        ])
    )

    if isfile(MODEL_PATH):
        print("Loading existing model from: {}".format(MODEL_PATH))
        model = torch.load(MODEL_PATH)
    else:
        print("New model created")

    return model


MODEL_PATH = "train_data/model.pt"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

def train():
    battle_features = td.get_player_features_from_games()
    print("training with: {}".format(DEVICE))
    model = get_model(10, 25, 2).to(DEVICE)
    print(model)
    model.train()

    opt = torch.optim.RMSprop(model.parameters())
    loss_func = nn.MSELoss()

    for game in battle_features:
        loss_game = []
        for features in game:
            input_features = torch.tensor(np.array(features[:-2]), dtype=torch.float).to(DEVICE)
            target_features = torch.tensor(np.array(features[-2:]), dtype=torch.float).to(DEVICE)
            opt.zero_grad()

            out = model(input_features)
            loss = loss_func(out, target_features)
            loss_game.append(loss)
            print("predict: {} target: {} loss: {}".format(out, target_features, loss))
            loss.backward()
            opt.step()
        if len(loss_game) != 0:
            print("average error: {}".format(sum(loss_game) / len(loss_game)))

    if isfile(ServerRecord.GAMES_FILE):
        os.remove(ServerRecord.GAMES_FILE)
    if isfile("train_data/game_player.npy"):
        os.remove("train_data/game_player.npy")

    torch.save(model, MODEL_PATH)

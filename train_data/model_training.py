import numpy as np
import torch.nn as nn
import torch
import train_data.data_import as td
from collections import OrderedDict

from torch.optim import SGD


def get_model(input_layer_size: int, hidden_layer_size: int, output_layer_size: int):
    model = nn.Sequential(
        OrderedDict([
            ("hidden_layer", nn.Linear(input_layer_size, hidden_layer_size)),
            ("activation", nn.ReLU()),
            ("output_layer", nn.Linear(hidden_layer_size, output_layer_size))
        ])
    )

    return model


MODEL_PATH = "train_data/model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 0.5

if __name__ == "__main__":
    battle_features = td.get_player_features_from_games()
    print("training with: {}".format(DEVICE))
    model = get_model(6, 10, 2).to(DEVICE)
    print(model)
    model.train()

    opt = SGD(model.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    for game in battle_features:
        for features in game:
            input_features = torch.tensor(np.array(features[:-2]), dtype=torch.float).to(DEVICE)
            target_features = torch.tensor(np.array(features[-2:]), dtype=torch.float).to(DEVICE)
            opt.zero_grad()

            out = model(input_features)
            loss = loss_func(out, target_features)
            print("predict: {} target: {} loss: {}".format(out, target_features, loss))
            loss.backward()
            opt.step()

    torch.save(model, MODEL_PATH)

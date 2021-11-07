import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence


class Predictor(nn.Module):
    """
    LSTM based predictor for the next 6 month averaged gas production.
    Parameters
    ----------
    feature_dim: int
        number of static features used to predict the gas production.
    sequence_dim: int
        dimension of an element of a sequence
    hidden_dim: int
        number of features in the hidden state h
    n_layers: int
        number of recurrent layers
    """
    def __init__(self, feature_dim=63, sequence_dim=4, hidden_dim=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=sequence_dim, hidden_size=hidden_dim,
                            num_layers=n_layers, batch_first=True)
        emb_size = feature_dim + hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )

    def forward(self, x):
        sequences, features = x['sequences'], x['features']
        _, (hidden, cell) = self.lstm(sequences.float())
        output = torch.cat([hidden[-1], features], dim=1)
        output = self.fc(output.float())
        return output

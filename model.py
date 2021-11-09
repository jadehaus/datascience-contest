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
    def __init__(self, feature_dim=55, sequence_dim=4, hidden_dim=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=sequence_dim, hidden_size=hidden_dim,
                            num_layers=n_layers, batch_first=True)
        emb_size = feature_dim + hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )
        # added by ljw / 20211108 - making SOS
        self.make_sos = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, sequence_dim)
        )

    def forward(self, x):

        sequences, features = x['sequences'], x['features'].float()

        if next(self.parameters()).is_cuda:
            sequences, features = sequences.cuda(), features.cuda()

        # with sos
        sos = self.make_sos(features).unsqueeze(1)  # [b 1 f]
        _, (hidden, cell) = self.lstm(sos)
        _, (hidden, cell) = self.lstm(sequences.float(), (hidden, cell))

        # without sos
        # _, (hidden, cell) = self.lstm(sequences.float())

        # readout
        output = torch.cat([hidden[-1], features], dim=1)
        output = self.fc(output.float())
        return output

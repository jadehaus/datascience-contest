import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import gurobipy as gp
from gurobipy import GRB
import csv

from torch.utils.data import DataLoader
from dataloader import exam_loader
import argument

class LSTMPredictor(nn.Module):
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
    def __init__(self, feature_dim=22, sequence_dim=4, hidden_dim=32, n_layers=3, args=None):
        super().__init__()
        self.args = args
        emb_size = feature_dim + hidden_dim
        self.make_sos = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            # nn.ReLU(),
            nn.Linear(feature_dim, sequence_dim)
        )
        self.gru = nn.GRU(input_size=sequence_dim, hidden_size=hidden_dim,
                          num_layers=n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )

    def forward(self, data):

        sequences, features = data

        if self.training:
            # add noise to feature
            features_noise = torch.randn_like(features) * features
            features = features + features_noise * float(self.args.noise)

            # add noise to sequence
            sequences_unpacked, lens_unpacked = pad_packed_sequence(sequences, batch_first=True)
            sequences_unpacked_noise = torch.randn_like(sequences_unpacked) * sequences_unpacked
            sequences_unpacked = sequences_unpacked + sequences_unpacked_noise * float(self.args.noise)
            sequences = pack_padded_sequence(sequences_unpacked, lens_unpacked, batch_first=True, enforce_sorted=False)

        sos = self.make_sos(features).unsqueeze(1)  # [b 1 f]
        _, hidden = self.gru(sos)
        _, hidden = self.gru(sequences.float(), hidden)

        output = torch.cat([hidden[-1], features], dim=1)
        output = self.fc(output.float())
        return output


class FeatureMLP(nn.Module):
    """
        2-layer MLP predictor for the first 6 month averaged gas production.
        Takes only static (time invariant) features to estimate the gas production.
        Parameters
        ----------
        feature_dim: int
            number of static features used to predict the gas production.
        emb_size: int
            number of features in the hidden layer
        n_layers: int
            number of recurrent layers
        """

    def __init__(self, feature_dim=22, emb_size=16, args=None):
        super().__init__()
        self.args = args

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, emb_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(emb_size, emb_size),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )


    def forward(self, data):

        sequences, features = data

        if self.training:
            # add noise to feature
            features_noise = torch.randn_like(features) * features
            features = features + features_noise * float(self.args.noise)

        # data = torch.cat((features, sequences), dim=1)
        output = self.fc(features)
        return output


class SequenceMLP(nn.Module):
    """
        2-layer MLP predictor for the last 6 month averaged gas production.
        Takes both the static features and seqeunces to estimate the gas production.
        Parameters
        ----------
        feature_dim: int
            number of static features used to predict the gas production.
        emb_size: int
            number of features in the hidden layer
        n_layers: int
            number of recurrent layers
        """

    def __init__(self, feature_dim=22, emb_size=128, args=None):
        super().__init__()
        sequence_dim = 90
        feature_dim += sequence_dim
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )

    def forward(self, data):

        sequences, features = data
        data = torch.cat((features, sequences), dim=1)

        output = self.fc(data.float())
        return output


class LPSolver:
    """
    LP solver that solves for the optimal choice of purchase of wells.
    Parameters
    ----------
    predictions: list
        list of predicted values of gas prod. for each well
    dataset: pandas.DataFrame
        test_dataset from the exam dataset csv file
    """
    def __init__(self, predictions, dataset):
        self.predictions = np.array(predictions)
        self.dataset = dataset

    def solver(self, s=5, b=1.5e7):
        """
        An LP solver for the given problem.
        Parameters
        ----------
        s: float
            shale gas price per 1mcf
        b: float
            total budget available in dollars
        Returns
        -------
        x: numpy.array of 0 or 1
            consists of solution for the purchase
        objVal: float
            the objective value for the profit
        """
        n = self.dataset.shape[0]
        c = self.dataset['Per Month Operation Cost ($)'].to_numpy()
        p = self.dataset['PRICE ($)'].to_numpy()
        a = self.predictions

        m = gp.Model()
        m.setParam('OutputFlag', 0)
        m.update()

        x = m.addMVar(shape=n, vtype=GRB.BINARY, name="x")
        m.setObjective((6*s*a - c - p) @ x, gp.GRB.MAXIMIZE)
        m.addConstr(p @ x <= b)
        m.optimize()

        return np.array(x.X).astype(int), m.objVal

    def export(self, file_path):
        """
        Exports csv file for the submission.
        Parameters
        ----------
        file_path: str
            directory for the csv file.
        """
        titles = ['Prediction', 'Purchase']
        prediction = self.predictions
        x, _ = self.solver()
        rows = np.array([prediction, x]).T

        with open(file_path, 'w') as f:
            write = csv.writer(f)
            write.writerow(titles)
            write.writerows(rows)

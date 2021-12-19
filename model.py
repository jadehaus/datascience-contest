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
import warnings


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
    def __init__(self, feature_dim=22, sequence_dim=4, hidden_dim=32, n_layers=2, noise=0):
        super().__init__()
        emb_size = hidden_dim + sequence_dim
        self.noise = noise
        self.feature_embedding = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, sequence_dim)
        )
        self.gru = nn.GRU(input_size=sequence_dim, hidden_size=hidden_dim, dropout=0.3,
                          num_layers=n_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(emb_size, 1)
        )

        # initialize params
        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        self.apply(weights_init)

    def forward(self, data):

        sequences, features = data

        if self.training:
            features_noise = torch.randn_like(features) * features
            features = features + features_noise * float(self.noise)

        sos = self.feature_embedding(features).unsqueeze(1)  # [b 1 f]
        _, hidden = self.gru(sos)
        _, hidden = self.gru(sequences.float(), hidden)

        emb_feat = self.feature_embedding(features)
        output = torch.cat([hidden[-1], emb_feat], dim=1)
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

    def __init__(self, feature_dim=22, emb_size=64, noise=0):
        super().__init__()
        self.noise = noise
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, emb_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(emb_size, 1)
        )

        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        self.apply(weights_init)

    def forward(self, data):

        _, features = data

        if self.training:
            features_noise = torch.randn_like(features) * features
            features = features + features_noise * float(self.noise)

        data = features
        output = self.fc(data.float())
        return output


class KnapsackSolver:
    """
    Knapsack solver that solves for the optimal choice of purchase of wells.
    Parameters
    ----------
    predictions: list
        list of expected predicted values of gas prod. for each well
    distribution: pandas.DataFrame
        predictive distribution of the predictions.
        used as the measure of uncertainty of the model.
    dataset: pandas.DataFrame
        test_dataset from the exam dataset csv file
    model: str
        specifies the model formulation of the problem.
    """
    def __init__(self, predictions, distribution, dataset, model='Expected Value'):
        self.predictions = np.array(predictions)
        self.distribution = distribution
        self.dataset = dataset
        self.model = model

    def solver(self, s=7, b=1.5e7, alpha=0.95):
        """
        An IP solver of a expectation-maximization knapsack problem
        with uncertainties for the given problem.
        Model is based on the work of [Peng & Zhang, 2012].
        Parameters
        ----------
        s: float
            shale gas price per 1mcf
        b: float
            total budget available in dollars
        alpha: float in [0,1]
            confidence level
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

        m = gp.Model()
        m.setParam('OutputFlag', 0)
        m.update()

        x = m.addMVar(shape=n, vtype=GRB.BINARY, name="x")
        m.addConstr(p @ x <= b)

        if self.model == 'Expected Value':
            pred = np.array(self.distribution.sum() / len(self.distribution))
            m.setObjective((6*s * pred - 6*c - p) @ x, gp.GRB.MAXIMIZE)
            m.optimize()

            solution = np.array(x.X).astype(int)
            objective = m.objVal

        elif self.model == 'Chance-Constrained':
            dist = self.distribution
            cdf = np.cumsum(dist) / np.sum(dist)
            idx = [np.argmin(np.abs(cdf[j]-(1-alpha))) for j in range(cdf.shape[1])]
            pred = np.array([dist[j].iloc[idx[j]] for j in range(cdf.shape[1])])

            v = m.addMVar(shape=1, name='v')
            m.addConstr((6*s * pred - 6*c - p) @ x >= v)
            m.setObjective(v, gp.GRB.MAXIMIZE)
            m.optimize()

            solution = np.array(x.X).astype(int)
            objective = m.objVal

        else:
            warnings.warn("Invalid model type. Processing simple predictions of optimization.", UserWarning)
            pred = self.predictions
            m.setObjective((6*s * pred - 6*c - p) @ x, gp.GRB.MAXIMIZE)
            m.optimize()

            solution = np.array(x.X).astype(int)
            objective = m.objVal

        return pred, solution, objective

    def export(self, file_path):
        """
        Exports csv file for the submission.
        Parameters
        ----------
        file_path: str
            directory for the csv file.
        """
        titles = ['Prediction', 'Purchase']
        pred, x, _ = self.solver()
        rows = np.array([pred, x]).T

        with open(file_path, 'w') as f:
            write = csv.writer(f)
            write.writerow(titles)
            write.writerows(rows)

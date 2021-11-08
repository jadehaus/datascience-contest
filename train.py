import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import os
import pathlib
import argparse
import yaml

from dataloader import seq_collate, preprocess
from model import Predictor
from datetime import datetime


def str_current_time():
    """
    Returns the current time in readable string.
    """
    now = datetime.now()
    current_datetime = \
        str(now.year) + \
        str(now.month) + \
        str(now.day) + \
        str(now.hour) + \
        str(now.strftime('%M')) +\
        str(now.strftime('%S'))

    return current_datetime


def process(model, data_loader, optimizer=None):
    """
    Process samples. If an optimizer is given, also train on those samples.
    Parameters
    ----------
    model : torch.nn.Module
        Model to train/evaluate.
    data_loader : torch.utils.data.DataLoader
        Pre-loaded dataset of training samples.
    optimizer : torch.optim (optional)
        Optimizer object. If not None, will be used for updating the model parameters.
    Returns
    -------
    mean_loss : float
        Mean MSE loss.
    """

    total_loss = 0
    n_data = len(data_loader)

    with torch.set_grad_enabled(optimizer is not None):
        for _, samples in enumerate(data_loader):
            data, label = samples, samples['target']
            prediction = model(data).view(-1)
            loss = F.mse_loss(prediction, label.float(), reduction='mean')
            total_loss += loss

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return total_loss / n_data


class Scheduler:
    """
    Custom scheduler class that early-stops the optimization process
    Parameters
    ----------
    patience: int
        a number of bad epochs that can be endured until early-stop
    """
    def __init__(self, patience=10):
        self.patience = patience
        self.num_bad_epoch = 0
        self.best_loss = 1e9

    def step(self, loss):
        """
        calculates a number of bad epochs and checks tolerences.
        Parameters
        ----------
        loss: float
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.num_bad_epoch = 0
        else:
            self.num_bad_epoch += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help='Print debug traces.',
        action='store_true',
    )
    args = parser.parse_args()

    loader_root = "./loader.yml"
    loader_config = yaml.load(open(loader_root, 'r'), Loader=yaml.SafeLoader)
    save_dir = os.path.join('./saved_params', str_current_time())
    os.makedirs(save_dir, exist_ok=True)

    # Setup Training Data
    train_root_path = "./datasets/trainSet.csv"
    test_root_path = "./datasets/examSet.csv"
    norm_dict = loader_config['norm_factor_dict']
    string_feats = loader_config['string_features']
    remove_feats = loader_config['remove_features']

    train_file = pd.read_csv(train_root_path)
    train_dataset, valid_dataset = preprocess(train_file, norm_dict, string_feats, remove_feats, ratio=0.2)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=seq_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True, collate_fn=seq_collate)

    print(len(train_dataset) + len(valid_dataset))
    # Import and train model
    max_epoch = 100
    model = Predictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = Scheduler(patience=10)

    for epoch in range(max_epoch):
        print()
        print(f"Epoch {epoch}...")
        train_loss = process(model, train_loader, optimizer=optimizer)
        valid_loss = process(model, valid_loader, optimizer=None)

        if args.debug:
            print(f"    Train loss: {train_loss}")
            print(f"    Valid loss: {valid_loss}")

        if True:
            print(f"    Train loss: {train_loss}")
            print(f"    Valid loss: {valid_loss}")

        scheduler.step(valid_loss)
        if scheduler.num_bad_epoch == 0:
            print(f"    Best model so far. Saving parameters...")
            torch.save(model.state_dict(), pathlib.Path(os.path.join(save_dir, "best_params.pkl")))
        elif scheduler.num_bad_epoch == scheduler.patience:
            break

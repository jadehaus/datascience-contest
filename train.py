import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import os
import pathlib
import argparse
import yaml

from dataloader import seq_collate, preprocess, exam_loader
from model import LSTMPredictor, FeatureMLP, SequenceMLP, LPSolver
from utilities import *


def process(model, data_loader, optimizer=None, device='cpu'):
    """
    Process samples. If an optimizer is given, also train on those samples.
    Parameters
    ----------
    model: torch.nn.Module
        Model to train/evaluate.
    data_loader: torch.utils.data.DataLoader
        Pre-loaded dataset of training samples.
    optimizer: torch.optim (optional)
        Optimizer object. If not None, will be used for updating the model parameters.
    device: torch.device
    Returns
    -------
    mean_loss : float
        Mean MSE loss.
    """
    total_loss = 0
    n_data = len(data_loader.dataset)

    with torch.set_grad_enabled(optimizer is not None):
        for _, samples in enumerate(data_loader):
            sequences = samples['sequences'].to(device)
            features = samples['features'].float().to(device)
            label = samples['target'].float()
            data = (sequences, features)

            prediction = model(data).view(-1).cpu() * 1e5
            loss = F.mse_loss(prediction, label, reduction='sum')
            total_loss += loss

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return total_loss / n_data


def train(model, optimizer, scheduler, dataset, save_dir,
          logfile=None, ratio=0.2, device='cpu'):
    """
    Trains the model given optimizera and scheduler.
    Returns the final parameter of the trained model.
    Parameters
    ----------
    model: torch.nn.Module
        Model to train.
    optimizer: torch.optim (optional)
        Optimizer object. If not None, will be used for updating the model parameters.
    scheduler: class
        A scheduler for training process.
    dataset: torch.utils.data.Dataset
        Pre-loaded dataset of training samples.
    save_dir: str
        Directory for saving model parameters.
    logfile: str (optional)
        Logfile directory for saving the logs.
    ratio: float in [0, 1]
        ratio for train and valid dataset split.
    device: torch.device
    Returns
    -------
    model.state_dict(): dict
    """
    # Printing model info and configure param file
    log(f"Model info: \n{model}", logfile, verbose=False)
    param_name = 'sequence' if dataset.has_sequence else 'feature'
    param_dir = pathlib.Path(os.path.join(save_dir, f"best_params_{param_name}.pkl"))

    # Split into train and valid set for feature dataset and sequence dataset respectively
    train_valid_split = [len(dataset) - int(len(dataset) * ratio), int(len(dataset) * ratio)]
    train_dataset, valid_dataset = random_split(dataset, train_valid_split)

    # Passing data to torch.utils.data.DataLoader, use seq_collate for LSTM models
    fn = seq_collate if model.__class__.__name__ == 'LSTMPredictor' else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=fn)

    for epoch in range(scheduler.max_epoch):
        log(f"EPOCH {epoch}...", logfile)
        train_loss = process(model, train_loader, optimizer=optimizer, device=device)
        valid_loss = process(model, valid_loader, optimizer=None, device=device)

        log(f"  Train loss: {train_loss.pow(0.5)}", logfile)
        log(f"  Valid loss: {valid_loss.pow(0.5)}", logfile)

        scheduler.step(valid_loss)
        if scheduler.num_bad_epoch == 0:
            log(f"  Best model so far. Saving parameters...", logfile)
            torch.save(model.state_dict(), param_dir)
        elif scheduler.num_bad_epoch == scheduler.patience:
            log(f"  {patience} epochs without improvement, early stopping", logfile)
            break

    model.load_state_dict(torch.load(param_dir))
    valid_loss = process(model, valid_loader, optimizer=None, device=device)
    log(f"  BEST VALID LOSS: {valid_loss.pow(0.5)}", logfile)

    return model.state_dict()


def evaluate(model, test_loader):
    """
    Evaluates the model and make predictions with data in test_loader.
    Parameters
    ----------
    model: torch.nn.Module
    test_loader: torch.utils.data.DataLoader
    Returns
    -------
    predictions: list
    """
    predictions = []
    with torch.set_grad_enabled(False):
        for _, samples in enumerate(test_loader):
            sequences = samples['sequences'].to(device)
            features = samples['features'].float().to(device)
            data = (sequences, features)

            production = model(data).view(-1).cpu().detach().numpy()
            predictions.extend(production * 1e5)

    return predictions


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help='Print debug traces.',
        action='store_true',
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=-1,
    )
    args = parser.parse_args()

    # Hyperparameters
    max_epoch = 500
    batch_size = 16
    ratio = 0.3
    lr = 1e-3
    patience = 20

    # Use augment option for more data, and pad option for LSTM models
    augment = False
    pad = False

    # Working directory setup
    loader_root = "./loader.yml"
    loader_config = yaml.load(open(loader_root, 'r'), Loader=yaml.SafeLoader)
    save_dir = os.path.join('./saved_params', str_current_time())

    # Debug argument setup
    if args.debug:
        save_dir = os.path.join('./saved_params', 'debug')
        max_epoch = 5
        patience = 1

    # cuda setup
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        device_num = int(float(args.gpu))
        torch.cuda.set_device(device_num)
        device = torch.device('cuda')

    # logging setup
    os.makedirs(save_dir, exist_ok=True)
    logfile = os.path.join(save_dir, 'train_log.txt')
    if os.path.exists(logfile):
        os.remove(logfile)

    log(f"Debug mode: {args.debug}", logfile)
    log(f"Max epochs: {max_epoch}", logfile)
    log(f"Batch size: {batch_size}", logfile)
    log(f"Learning rate: {lr}", logfile)
    log(f"Device: {device}")

    # Setup Training Data
    train_root_path = "./datasets/trainSet.csv"
    test_root_path = "./datasets/examSet.csv"
    norm_dict = loader_config['norm_factor_dict']
    removes = loader_config['remove_features']

    log(f"Initiating data augmentation...", logfile)
    train_file = pd.read_csv(train_root_path)
    feature_dataset, sequence_dataset = preprocess(train_file, norm_dict, removes, augment=augment, pad=pad)
    log(f"Data loading completed. "
        f"{len(feature_dataset)} total feature data and "
        f"{len(sequence_dataset)} sequence data.", logfile)
    log(f"Configuration Information: \n{loader_config}", logfile, verbose=False)

    # Import and train model for feature data
    model_feature = FeatureMLP().to(device)
    log(f"Training {model_feature.__class__.__name__} for feature data", logfile)
    optimizer = torch.optim.Adam(model_feature.parameters(), lr=lr)
    scheduler = Scheduler(patience=patience, max_epoch=max_epoch)
    train(model_feature, optimizer, scheduler, feature_dataset,
          save_dir, logfile=logfile, ratio=ratio, device=device)

    # Import and train model for sequence data
    model_sequence = SequenceMLP().to(device)
    log(f"Training {model_sequence.__class__.__name__} for sequence data", logfile)
    optimizer = torch.optim.Adam(model_sequence.parameters(), lr=lr)
    scheduler = Scheduler(patience=patience, max_epoch=max_epoch)
    train(model_sequence, optimizer, scheduler, sequence_dataset,
          save_dir, logfile=logfile, ratio=ratio, device=device)

    # Make predictions with the exam data
    test_file = pd.read_csv(test_root_path)
    feature_test, sequence_test = exam_loader(train_file, test_file, norm_dict, removes, pad=pad)

    # Passing data to torch.utils.data.DataLoader, use seq_collate for LSTM models
    fn_ft = seq_collate if model_feature.__class__.__name__ == 'LSTMPredictor' else None
    fn_sq = seq_collate if model_sequence.__class__.__name__ == 'LSTMPredictor' else None
    feature_test = DataLoader(feature_test, batch_size=batch_size, shuffle=False, collate_fn=fn_ft)
    sequence_test = DataLoader(sequence_test, batch_size=batch_size, shuffle=False, collate_fn=fn_sq)

    feature_predictions = evaluate(model_feature, feature_test)
    sequence_predictions = evaluate(model_sequence, sequence_test)

    predictions = feature_predictions + sequence_predictions
    submission_file = os.path.join(save_dir, 'submission.csv')
    solver = LPSolver(predictions, test_file)
    solver.export(submission_file)
    log(f'Submission file successfully exported to {submission_file}', logfile)

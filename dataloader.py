import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


def seq_tensor(gas, cnd, hrs, n=(1e5, 1e4, 1e3)):
    """
    Reformats three given seqeunces of gas, cnd production and hrs
    into a single sequence of vectors of dimension 4.
    Parameters
    ----------
    gas, cnd, hrs: array_like, array_like, array_like
        array of gas prod. values, cnd prod. values, and prod. hrs, respectively.
    n: tuple
        consists of normalizing factor for each sequence
    """
    if np.isnan(hrs[0]):
        return torch.zeros(1, 4)

    rest, sequences = 0, []
    for j, h in enumerate(hrs):
        if h == 0:
            rest += 1
        else:
            sequences.append([gas[j]/n[0], cnd[j]/n[1], hrs[j]/n[2], rest])
            rest = 0

    return torch.tensor(sequences)


def seq_collate(batch):
    """
    A custom collate function that packs sequences and stacks samples in a batch.
    The sequences are stored as Packed Sequence objects.
    Parameters
    ----------
    batch: list
        contains torch.utils.data.Data objects
    Returns
    -------
    padded_batch: dict
        dictionary of features, sequences and target values
    """
    features = [sample['features'] for sample in batch]
    sequences = [sample['sequences'] for sample in batch]
    targets = [sample['target'] for sample in batch]

    lengths = [len(s) for s in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True).contiguous()
    packed_sequences = pack_padded_sequence(padded_sequences, lengths, batch_first=True, enforce_sorted=False)
    padded_batch = {'features': torch.stack(features).contiguous(),
                    'sequences': packed_sequences,
                    'target': torch.stack(targets).contiguous()}

    return padded_batch


def preprocess(dataset, normalize_dict, string_feats, remove_feats=None, ratio=0.2):
    """
    Preprocesses datasets via normalizing and removing unnecessary features.
    Also one-hot-encodes string features.
    Parameters
    ----------
    dataset: pandas.DataFrame
    normalize_dict: dict
    string_feats: list
    remove_feats: (optional) list
    ratio: float in [0, 1]
        fraction value for the size of validation dataset
    Returns
    -------
    train_dataset: WellDataset
    valid_dataset: WellDataset
    """
    if remove_feats is not None:
        dataset = dataset.drop(remove_feats, axis=1)

    dataset = pd.get_dummies(dataset, columns=string_feats)
    for feats in normalize_dict:
        dataset[feats] /= float(normalize_dict[feats])

    total_features = [f for f in dataset.columns if ('MONTH' not in f and 'mo.' not in f)]
    valid_data = dataset.sample(frac=ratio).reset_index(drop=True)
    train_data = dataset.drop(valid_data.index).reset_index(drop=True)

    train_dataset = WellDataset(train_data, total_features, train=True)
    valid_dataset = WellDataset(valid_data, total_features, train=False)

    return train_dataset, valid_dataset


class WellDataset(Dataset):
    """
    A dataset class that inherits torch.utils.data.Dataset.
    Parameters
    ----------
    dataset: pandas.DataFrame
    features: list
    gas_norm: float (optional)
    train: bool (optional)
    """
    def __init__(self, dataset, features, gas_norm=1e5, train=True):
        self.dataset = dataset
        self.features = features
        self.train = train
        self.gas_norm = gas_norm

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        mth = 37 if self.train else 31
        gas = self.dataset[[f'GAS_MONTH_{j}' for j in range(1, mth)]].loc[idx]
        cnd = self.dataset[[f'CND_MONTH_{j}' for j in range(1, mth)]].loc[idx]
        hrs = self.dataset[[f'HRS_MONTH_{j}' for j in range(1, mth)]].loc[idx]

        sequences = seq_tensor(gas, cnd, hrs)
        empty_sequence = (len(sequences) == 1)
        static_features = torch.tensor(self.dataset[self.features].loc[idx])

        target_name = 'First 6 mo. Avg. GAS (Mcf)' if empty_sequence else 'Last 6 mo. Avg. GAS (Mcf)'
        target = torch.tensor(self.dataset[target_name].loc[idx]/self.gas_norm)

        sample = {'features': static_features, 'sequences': sequences, 'target': target}

        return sample

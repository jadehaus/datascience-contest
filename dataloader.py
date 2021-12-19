import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.impute import SimpleImputer


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

    rest, sequences = 0, [[0, 0, 0, 0]]
    for j, h in enumerate(hrs):
        if h == 0:
            rest += 1
        else:
            gas_norm, cnd_norm, hrs_norm = gas[j]/n[0], cnd[j]/n[1], hrs[j]/n[2]
            rates = [gas_norm/hrs_norm, cnd_norm/hrs_norm, hrs_norm, rest]
            sequences.append(rates)
            rest = 0

    if not sequences:
        return torch.zeros(1, 4)

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


def preprocess(dataset, normalize_dict, remove_feats=None, augment=False):
    """
    Preprocesses datasets via normalizing and removing unnecessary features.
    Also one-hot-encodes string features.
    Parameters
    ----------
    dataset: pandas.DataFrame
    normalize_dict: dict
    remove_feats: (optional) list
    augment: (optional) bool
        augments and reproduces data if true
    Returns
    -------
    feature_dataset: WellDataset
    sequence_dataset: WellDataset
    """
    if remove_feats is not None:
        dataset = dataset.drop(remove_feats, axis=1)

    # types of features
    string_feats = dataset.select_dtypes(include='object').columns.tolist()
    value_feats = dataset.select_dtypes(include=np.number).columns.tolist()
    value_feats = [f for f in value_feats if ('MONTH' not in f and 'mo.' not in f)]

    # maps the direction to an angle
    direction = 'Bot-Hole direction (N/S)/(E/W)'
    dataset[direction] = np.arctan(dataset[direction])

    # imputes missing data and one-hot encodes if necessary
    if value_feats:
        num_imputer = SimpleImputer(strategy='mean')
        dataset[value_feats] = num_imputer.fit_transform(dataset[value_feats])
    if string_feats:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        dataset[string_feats] = cat_imputer.fit_transform(dataset[string_feats])
        dataset = pd.get_dummies(dataset, columns=string_feats)

    # normalize data
    for feats in value_feats:
        dataset[feats] /= float(normalize_dict[feats])

    # split into data w/ or w/o sequence data
    target_name = 'Last 6 mo. Avg. GAS (Mcf)'
    feature_dataset = dataset
    sequence_dataset = dataset.dropna(subset=[target_name]).reset_index(drop=True)

    # data augmentation / inplace addition of data
    if augment:
        sequence_dataset = augment_data(sequence_dataset)

    total_features = [f for f in dataset.columns if ('MONTH' not in f and 'mo.' not in f)]
    sequence_dataset = WellDataset(sequence_dataset, total_features, sequence=True)
    feature_dataset = WellDataset(feature_dataset, total_features, sequence=False)

    return feature_dataset, sequence_dataset


def exam_loader(train_data, exam_data, norm_dict, remove_feats=None):
    """
    Preprocesses exam datasets via normalizing and removing unnecessary features.
    Also one-hot-encodes string features.
    Parameters
    ----------
    train_data: pandas.DataFrame
        This is required in order to align the dummy values.
    exam_data: pandas.DataFrame
    norm_dict: dict
    remove_feats: (optional) list
    Returns
    -------
    exam_dataset: WellDataset
    """
    if remove_feats is not None:
        train_data = train_data.drop(remove_feats, axis=1)
        exam_data = exam_data.drop(remove_feats, axis=1)

    train_index = train_data.index
    dataset = pd.concat([train_data, exam_data]).reset_index(drop=True)

    # remove decision related features
    decision_feats = [col for col in dataset.columns if '$' in col]
    dataset = dataset.drop(decision_feats, axis=1)

    # types of features
    string_feats = dataset.select_dtypes(include='object').columns.tolist()
    value_feats = dataset.select_dtypes(include=np.number).columns.tolist()
    value_feats = [f for f in value_feats if ('MONTH' not in f and 'mo.' not in f)]

    # maps the direction to an angle
    direction = 'Bot-Hole direction (N/S)/(E/W)'
    dataset[direction] = np.arctan(dataset[direction])

    # imputes missing data and one-hot encodes if necessary
    if value_feats:
        num_imputer = SimpleImputer(strategy='mean')
        dataset[value_feats] = num_imputer.fit_transform(dataset[value_feats])
    if string_feats:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        dataset[string_feats] = cat_imputer.fit_transform(dataset[string_feats])
        dataset = pd.get_dummies(dataset, columns=string_feats)

    # normalize data
    for feats in value_feats:
        dataset[feats] /= float(norm_dict[feats])

    # remove train dataset back again
    dataset = dataset.drop(train_index).reset_index(drop=True)

    # split into data w/ or w/o sequence data
    target_name = 'GAS_MONTH_1'
    feature_dataset = dataset[dataset[target_name].isna()]
    sequence_dataset = dataset.dropna(subset=[target_name]).reset_index(drop=True)

    total_features = [f for f in dataset.columns if ('MONTH' not in f and 'mo.' not in f)]
    exam_feature_dataset = WellDataset(feature_dataset, total_features, sequence=False, exam=True)
    exam_sequence_dataset = WellDataset(sequence_dataset, total_features, sequence=True, exam=True)

    return exam_feature_dataset, exam_sequence_dataset


def augment_data(dataset):
    """
    Append additional data rows to the pandas.dataframe,
    where the new row consists of partial subsequences of gas, cnd, and hrs.
    Parameters
    ----------
    dataset: pandas.DataFrame
    Returns
    -------
    dataset: pandas.DataFrame
    """
    target = 'Last 6 mo. Avg. GAS (Mcf)'
    gas_sequences = [f'GAS_MONTH_{j}' for j in range(1, 37)]
    sequences = dataset.dropna(subset=gas_sequences).reset_index(drop=True)

    # Appending additional subsequences to the rows
    data_list = []
    for idx in range(sequences.shape[0]):
        for t in range(1, 30):

            # Do not append when the production stopped for at least a month
            target_hrs = [f'HRS_MONTH_{j}' for j in range(t+1, t+7)]
            if 0 in list(sequences.iloc[idx][target_hrs]):
                continue

            row = sequences.copy().iloc[idx]
            hrs = [f'HRS_MONTH_{j}' for j in range(t+1, 37)]
            gas = [f'GAS_MONTH_{j}' for j in range(t+1, 37)]
            cnd = [f'CND_MONTH_{j}' for j in range(t+1, 37)]

            # Changing the target value for the last 6 months
            target_gas = [f'GAS_MONTH_{j}' for j in range(t+1, t+7)]
            row[target] = sum(row[target_gas]) / len(target_gas)
            row[hrs], row[gas], row[cnd] = 0, 0, 0

            data_list.append(dict(row))

    df = pd.DataFrame(data_list)
    dataset = pd.concat([dataset, df], axis=0, ignore_index=True)
    dataset.reset_index(drop=True)

    return dataset


class WellDataset(Dataset):
    """
    A dataset class that inherits torch.utils.data.Dataset.
    Parameters
    ----------
    dataset: pandas.DataFrame
    features: list
    gas_norm: float (optional)
    sequence: bool (optional)
    exam: bool (optional)
    """
    def __init__(self, dataset, features, gas_norm=1, sequence=False, exam=False):
        self.dataset = dataset
        self.features = features
        self.gas_norm = gas_norm
        self.has_sequence = sequence
        self.exam = exam

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        gas = np.array(self.dataset[[f'GAS_MONTH_{j}' for j in range(1, 31)]].loc[idx])
        cnd = np.array(self.dataset[[f'CND_MONTH_{j}' for j in range(1, 31)]].loc[idx])
        hrs = np.array(self.dataset[[f'HRS_MONTH_{j}' for j in range(1, 31)]].loc[idx])

        sequences = seq_tensor(gas, cnd, hrs) if self.has_sequence else torch.zeros(1, 4)
        target = torch.zeros(1)
        if not self.exam:
            target_name = 'Last 6 mo. Avg. GAS (Mcf)' if self.has_sequence else 'First 6 mo. Avg. GAS (Mcf)'
            target = torch.tensor(self.dataset[target_name].loc[idx]/self.gas_norm)

        static_features = torch.tensor(self.dataset[self.features].loc[idx])
        sample = {'features': static_features, 'sequences': sequences, 'target': target}

        return sample

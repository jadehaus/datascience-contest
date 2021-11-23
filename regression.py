import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataloader import seq_collate, preprocess, exam_loader
from utilities import *

import os
import pathlib
import argparse
import yaml
import joblib

from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import VotingRegressor, StackingRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore')


def model_perf(model, data, label, prefix='', logfile=None):
    pred = model.predict(data)
    r2 = r2_score(label, pred)
    rmse = mean_squared_error(label, pred, squared=False)
    log(f'{prefix} R^2 score = {r2:.3f}, RMSE = {rmse:.4f}', logfile)


def model_fit_perf(model, data, label, prefix='', logfile=None):
    train_data, valid_data, train_label, valid_label = train_test_split(data, label, test_size=0.2, random_state=0)
    model.fit(train_data, train_label)
    log(prefix, logfile)
    model_perf(model, train_data, train_label, ' Training ')
    model_perf(model, valid_data, valid_label, ' Validation')


def fit_regression_models(data, label, save_dir, logfile=None, sequence=False, exp=True):

    ridge = Ridge()
    bagging = BaggingRegressor()
    random_forest = RandomForestRegressor()
    xgb = XGBRegressor()
    lgbm = LGBMRegressor()

    param_ridge = {'alpha': [a * 0.01 for a in range(1000)]}
    param_bagging = {'max_features': [0.5, 0.6, 0.7, 0.8, 0.9],
                     'max_samples': [0.5, 0.6, 0.7, 0.8, 0.9],
                     'n_estimators': [10, 15, 20, 25, 30]}
    param_random_forest = {'max_depth': [2, 3, 5, 7, 10],
                           'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40]}
    param_xgboost = {'max_depth': [2, 3, 5, 7],
                     'n_estimators': [5, 10, 12, 15],
                     'subsample': [0.3, 0.4, 0.5, 0.6]}
    param_lightgbm = {'max_depth': [2, 5, 7, 10, 15],
                      'n_estimators': [20, 25, 30, 40, 45, 50],
                      'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.8]}

    target = np.log(label) if exp else label

    search = GridSearchCV(ridge, param_ridge)
    search.fit(data, target)
    log(f'Best params: {search.best_params_}', logfile)
    ridge = Ridge(**search.best_params_)
    model_fit_perf(ridge, data, target, "Ridge", logfile=logfile)

    search = GridSearchCV(bagging, param_bagging)
    search.fit(data, target)
    log(f'Best params: {search.best_params_}', logfile)
    bagging = BaggingRegressor(**search.best_params_)
    model_fit_perf(bagging, data, target, "Bagging", logfile=logfile)

    search = GridSearchCV(random_forest, param_random_forest)
    search.fit(data, target)
    log(f'Best params: {search.best_params_}', logfile)
    random_forest = RandomForestRegressor(**search.best_params_)
    model_fit_perf(random_forest, data, target, "Random Forest", logfile=logfile)

    search = GridSearchCV(xgb, param_xgboost)
    search.fit(data, target)
    log(f'Best params: {search.best_params_}', logfile)
    xgb = XGBRegressor(**search.best_params_)
    model_fit_perf(xgb, data, target, "XGBoost", logfile=logfile)

    search = GridSearchCV(lgbm, param_lightgbm)
    search.fit(data, target)
    log(f'Best params: {search.best_params_}', logfile)
    lgbm = LGBMRegressor(**search.best_params_)
    model_fit_perf(lgbm, data, target, "LightGBM", logfile=logfile)

    estimators = [('Bagging', bagging), ('Random Forest', random_forest),
                  ('XGBoost', xgb), ('LightGBM', lgbm), ('ridge', ridge)]

    voting_regressor = VotingRegressor(estimators)
    model_fit_perf(voting_regressor, data, target, "Voting(mean) Ensemble", logfile=logfile)

    # Final model selection and making predictions
    pred = voting_regressor.predict(data)
    pred = np.exp(pred) if exp else pred
    log(f' Total RMSE for final model: {np.sqrt(np.sum(np.square(label - pred)) / len(pred))}', logfile)

    # Saves parameter
    param_name = 'sequence' if sequence else 'feature'
    param_dir = os.path.join(save_dir, f"best_params_{param_name}.pkl")
    joblib.dump(voting_regressor, param_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help='Print debug traces.',
        action='store_true',
    )
    args = parser.parse_args()

    # Working directory setup
    loader_root = "./loader.yml"
    loader_config = yaml.load(open(loader_root, 'r'), Loader=yaml.SafeLoader)
    save_dir = os.path.join('./saved_params/regression', str_current_time())

    # Debug argument setup
    if args.debug:
        save_dir = os.path.join('./saved_params/regression', 'debug')

    # logging setup
    os.makedirs(save_dir, exist_ok=True)
    logfile = os.path.join(save_dir, 'train_log.txt')
    if os.path.exists(logfile):
        os.remove(logfile)

    # Feature setup
    train_root_path = "./datasets/trainSet.csv"
    norm_dict = loader_config['norm_factor_dict']
    remove_feats = loader_config['remove_features']
    last_prod = 'Last 6 mo. Avg. GAS (Mcf)'
    first_prod = 'First 6 mo. Avg. GAS (Mcf)'
    dataset = pd.read_csv(train_root_path)

    # Preprocess data
    dataset = dataset.drop(remove_feats, axis=1)
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
    for feats in norm_dict:
        dataset[feats] /= float(norm_dict[feats])

    # split into data w/ or w/o sequence data
    feature_dataset = dataset[dataset[last_prod].isna()]
    sequence_dataset = dataset.dropna(subset=[last_prod]).reset_index(drop=True)

    gas = np.array(sequence_dataset[[f'GAS_MONTH_{j}' for j in range(1, 31)]])
    cnd = np.array(sequence_dataset[[f'CND_MONTH_{j}' for j in range(1, 31)]])
    hrs = np.array(sequence_dataset[[f'HRS_MONTH_{j}' for j in range(1, 31)]])

    log('Fitting sequence data', logfile)
    data = gas
    label = np.array(sequence_dataset[last_prod])
    fit_regression_models(data, label, save_dir, logfile=logfile, sequence=True, exp=False)

    log('Fitting static feature data', logfile)
    total_features = [f for f in feature_dataset if ('MONTH' not in f and 'mo.' not in f)]
    data = np.array(feature_dataset[total_features])
    label = np.array(feature_dataset[first_prod])
    fit_regression_models(data, label, save_dir, logfile=logfile, sequence=False, exp=False)

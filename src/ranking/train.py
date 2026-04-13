import numpy as np
import pandas as pd
from lightgbm import LGBMRanker


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, qid_train: pd.DataFrame, params: dict):
    '''

    Trains LGBM Ranker model and tracks performance metrics using MLFlow

    :param X_train: containing pairwise feature matrix.
    :param y_train pd.DataFrame containing relevance labels.
    :param qid: pd.DataFrame containing query ids, one per row in X/y.
    :param params: dictionary containing parameters for XGBoost model
    :return: model, metrics,
    '''

    model = LGBMRanker(**params)

    # Train model
    model = LGBMRanker(**params)

    y_train = np.asarray(y_train).reshape(-1)
    qids = np.asarray(qid_train).reshape(-1)
    group_train = pd.Series(qids).value_counts(sort=False).sort_index().tolist()

    model.fit(X_train, y_train, group=group_train)

    return model


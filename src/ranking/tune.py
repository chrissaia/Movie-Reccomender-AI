import time
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgbm
from sklearn.model_selection import GroupKFold

from src.ranking.evaluate import ndcg_at_k


def _make_group_sizes(qids: np.ndarray) -> list[int]:
    """
    Convert per-row query ids into LightGBM group sizes.

    Example:
    qids = [0,0,0,1,1,2,2,2]
    returns [3,2,3]
    """
    return pd.Series(qids).value_counts(sort=False).sort_index().tolist()


def tune_model(X, y, qid, n_trials: int = 10, n_splits: int = 3, seed: int = 42):
    """
    Tune a LightGBM LambdaRank models with Optuna using GroupKFold CV.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : array-like
        Relevance labels
    qid : array-like
        Query/group ids
    n_trials : int
        Number of Optuna trials
    n_splits : int
        Number of GroupKFold splits
    seed : int
        Random seed

    Returns
    -------
    dict
        Best hyperparameters
    """

    # defensive conversion
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    else:
        X = X.copy()

    y = np.asarray(y).reshape(-1)
    qid = np.asarray(qid).reshape(-1)

    if len(X) - len(y) > 1 or len(X) - len(qid) > 1:
        raise ValueError("X, y, and qid must have the same length")

    gkf = GroupKFold(n_splits=n_splits)

    def objective(trial):
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 250),
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": -1,
        }

        fold_scores = []


        for train_idx, val_idx in gkf.split(X, y, groups=qid):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            qid_train, qid_val = qid[train_idx], qid[val_idx]

            group_train = _make_group_sizes(qid_train)
            group_val = _make_group_sizes(qid_val)

            model = lgbm.LGBMRanker(**params)
            model.fit(
                X_train,
                y_train,
                group=group_train,
                eval_set=[(X_val, y_val)],
                eval_group=[group_val],
                callbacks=[lgbm.early_stopping(10, verbose=False)],
            )

            val_scores = model.predict(X_val)

            fold_ndcg = ndcg_at_k(y_val, val_scores, qid_val, k=10)
            fold_scores.append(fold_ndcg)

        return float(np.mean(fold_scores))

    start = time.time()

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    time_taken = time.time() - start

    print("Time taken:", round(time_taken, 2), "seconds")
    print("Best Params:", study.best_params)
    print("Best CV NDCG@10:", study.best_value)

    return study.best_params
import numpy as np
import pandas as pd

def group_train_test_split(X: pd.DataFrame, y: pd.DataFrame, qid: pd.DataFrame, test_size=0.2, random_state=42):
    """

    I needed to make my own function for train test split because of qid

    :param X: training data
    :param y: label
    :param qid: # WHAT IS QID?
                # Each movie_a is its own qid and is then tested against every other movie
                EXAMPLE:
                movie_a 	movie_b 	features	label	qid
                -----------------------------------------------
                Shining	    Star Wars	   ...	      2	     0
                Shining	    Airplane!	   ...	      0	     0
                Star Wars	Shining	       ...	      2	     1
                Star Wars	Airplane!	   ...	      1	     1
    :param test_size: the split for train/test ex. 0.2 means 80% train / 20% test split
    :param random_state: seed for numpy

    :return: X_train, X_test, y_train, y_test, qid_train, qid_test
    """
    np.random.seed(random_state)

    # unique queries (movies)
    unique_qids = np.unique(qid)

    # shuffle queries
    shuffled_qids = np.random.permutation(unique_qids)

    # split point
    split_idx = int(len(shuffled_qids) * (1 - test_size))

    train_qids = shuffled_qids[:split_idx]
    test_qids = shuffled_qids[split_idx:]

    # masks
    train_mask = np.isin(qid, train_qids)
    test_mask = np.isin(qid, test_qids)

    # split train
    X_train = X[train_mask]
    y_train = y[train_mask]
    qid_train = qid[train_mask]

    # split test
    X_test = X[test_mask]
    y_test = y[test_mask]
    qid_test = qid[test_mask]

    return X_train, X_test, y_train, y_test, qid_train, qid_test
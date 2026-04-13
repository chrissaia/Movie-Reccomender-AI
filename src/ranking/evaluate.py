import numpy as np
import pandas as pd
import time

def ndcg_at_k(y_true, y_score, qids, k=10):
    df_eval = pd.DataFrame({
        "y_true": y_true,
        "y_score": y_score,
        "qid": qids
    })

    ndcgs = []

    for _, group in df_eval.groupby("qid"):
        group = group.sort_values("y_score", ascending=False).head(k)
        gains = (2 ** group["y_true"].to_numpy() - 1)
        discounts = np.log2(np.arange(2, len(group) + 2))
        dcg = np.sum(gains / discounts)

        ideal = group.sort_values("y_true", ascending=False).head(k)
        ideal_gains = (2 ** ideal["y_true"].to_numpy() - 1)
        ideal_discounts = np.log2(np.arange(2, len(ideal) + 2))
        idcg = np.sum(ideal_gains / ideal_discounts)

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs))

def evaluate_model(model, X_test, y_test, qid_test):
    '''
    Evaluate models performance on the test set

    :param model:
    :param X_test:
    :param y_test:
    :param qid_test:

    :returns
    model_scores, model_metrics
    '''

    start_time = time.time()
    scores = model.predict(X_test)
    predict_time = time.time() - start_time

    qid_test = np.asarray(qid_test).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)
    scores = np.asarray(scores).reshape(-1)

    lgbm_ndcg = ndcg_at_k(y_test, scores, qid_test, k=10)

    metrics = {
        "predict_time": predict_time,
        "lgbm_ndcg_10:": lgbm_ndcg,
    }

    return scores, metrics
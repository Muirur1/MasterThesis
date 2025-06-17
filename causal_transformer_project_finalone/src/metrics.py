import numpy as np

def weighted_mse(y_true, y_pred, weight=None):
    if weight is None:
        return np.mean((y_true - y_pred) ** 2)
    return np.average((y_true - y_pred) ** 2, weights=weight)

def weighted_rmse(y_true, y_pred, weight=None):
    """
    Root Mean Squared Error for factual outcomes.
    """
    return np.sqrt(weighted_mse(y_true, y_pred, weight))

def weighted_pehe(ite_true, ite_pred, weight=None):
    """
    Precision in Estimation of Heterogeneous Effects.
    """
    if weight is None:
        pehe = np.mean((ite_pred - ite_true) ** 2)
    else:
        pehe = np.average((ite_pred - ite_true) ** 2, weights=weight)
    return np.sqrt(pehe)

def weighted_ate(ite_true, ite_pred, weight=None):
    """
    Absolute difference in Average Treatment Effects.
    """
    if weight is None:
        return abs(np.mean(ite_pred) - np.mean(ite_true))
    return abs(np.average(ite_pred, weights=weight) - np.average(ite_true, weights=weight))

def weighted_policy_risk(ite_true, ite_pred, weight=None):
    """
    Policy Risk measures the proportion of incorrect treatment decisions.
    """
    treated_policy = (ite_pred > 0).astype(int)
    optimal_policy = (ite_true > 0).astype(int)
    incorrect = (treated_policy != optimal_policy).astype(int)
    if weight is None:
        return np.mean(incorrect)
    return np.average(incorrect, weights=weight)

def compute_all_metrics_full(
    ite_true, ite_pred,
    factual_true=None, factual_pred=None,
    weight=None,
    return_dict=True
):
    """
    Returns dictionary or list of metrics:
    - PEHE
    - ATE
    - RMSE (if factual outcomes are available)
    - Policy Risk
    """
    metrics = {
        "PEHE": weighted_pehe(ite_true, ite_pred, weight),
        "ATE": weighted_ate(ite_true, ite_pred, weight),
        "Policy Risk": weighted_policy_risk(ite_true, ite_pred, weight),
    }

    if factual_true is not None and factual_pred is not None:
        metrics["RMSE"] = weighted_rmse(factual_true, factual_pred, weight)
    else:
        metrics["RMSE"] = np.nan

    return metrics if return_dict else list(metrics.values())

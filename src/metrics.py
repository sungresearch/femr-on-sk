import numpy as np
import pandas as pd


def _binary_clf_curve(y_true, y_score, pos_label=1, sample_weight=None):
    """Calculate true and false positives per binary classification threshold.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
    pos_label : int or str, default=None
        The label of the positive class
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """

    y_true = y_true == pos_label

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.0

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        fps = np.cumsum(weight)[threshold_idxs] - tps
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]


def precision_recall_curve(
    y_true, y_pred, pos_label=None, sample_weight=None, pi0=None
):
    """Compute precision-recall (with optional calibration) pairs for different probability thresholds
    This implementation is a modification of scikit-learn "precision_recall_curve" function that adds calibration
    ----------
    y_true : array, shape = [n_samples]
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.
    probas_pred : array, shape = [n_samples]
        Estimated probabilities or decision function.
    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    Returns
    -------
    calib_precision : array, shape = [n_thresholds + 1]
        Calibrated Precision values such that element i is the calibrated precision of
        predictions with score >= thresholds[i] and the last element is 1.
    recall : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.
    thresholds : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
        Increasing thresholds on the decision function used to compute
        precision and recall.
    """

    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight
    )

    if pi0 is not None:
        pi = np.sum(y_true) / float(np.array(y_true).shape[0])
        ratio = pi * (1 - pi0) / (pi0 * (1 - pi))
        precision = tps / (tps + ratio * fps)
    else:
        precision = tps / (tps + fps)

    precision[np.isnan(precision)] = 0

    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def average_precision_score_calibrated(
    y_true: np.array,
    y_pred: np.array,
    pos_label: int = 1,
    sample_weight: np.array = None,
    pi0: float = 0.5,
):
    precision, recall, _ = precision_recall_curve(
        y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight, pi0=pi0
    )

    return -np.sum(np.diff(recall) * np.array(precision)[:-1])


def expected_calibration_error(
    y_true: np.array,
    y_pred: np.array,
    num_bins: int = 10,
    metric_variant: str = "abs",
    quantile_bins: bool = False,
    correct_y_pred: bool = False,
    b0: float = None,
):
    """
    Computes the calibration error with a binning estimator over equal sized bins
    See http://arxiv.org/abs/1706.04599 and https://arxiv.org/abs/1904.01685.
    """
    if metric_variant == "abs":
        transform_func = np.abs
    elif (metric_variant == "squared") or (metric_variant == "rmse"):
        transform_func = np.square
    elif metric_variant == "signed":
        transform_func = identity
    else:
        raise ValueError("provided metric_variant not supported")

    if quantile_bins:
        cut_fn = pd.qcut
    else:
        cut_fn = pd.cut

    if correct_y_pred and b0 is not None:
        y_pred = correct_risk(y_pred, b=b0, b1=np.mean(y_true))

    bin_ids = cut_fn(y_pred, num_bins, labels=False, retbins=False, duplicates="drop")
    df = pd.DataFrame({"pred_probs": y_pred, "labels": y_true, "bin_id": bin_ids})
    ece_df = (
        df.groupby("bin_id")
        .agg(
            pred_probs_mean=("pred_probs", "mean"),
            labels_mean=("labels", "mean"),
            bin_size=("pred_probs", "size"),
        )
        .assign(
            bin_weight=lambda x: x.bin_size / df.shape[0],
            err=lambda x: transform_func(x.pred_probs_mean - x.labels_mean),
        )
    )
    result = np.average(ece_df.err.values, weights=ece_df.bin_weight)
    if metric_variant == "rmse":
        result = np.sqrt(result)
    return result


def correct_risk(
    p: np.ndarray,
    b: float,
    b1: float,
) -> np.ndarray:
    """
    correct probability given new base rate according to https://dl.acm.org/doi/10.5555/1642194.1642224
    use case:
        sometimes during training the base rate of target distribution is unknown or when there are constraints (e.g.,
        few shots learning), and so balanced sampling might be used to create training set. This over/under sampling
        can impact calibration (over/under-estimating risk) when the model is applied to a target dataset with a
        different baserate. This function corrects the predicted risk based on the new, known base rate of the target
        distribution.
    ----------
    p : array
        Predicted risks
    b : float
        base rate of events on which the model that produced p was trained
    b1: float
        new base rate of events to which the model is being applied
    Returns
    -------
    p1: array
        corrected predicted risks
    """

    return b1 * ((p - p * b) / (b - p * b + b1 * p - b * b1))


def identity(x):
    """
    Returns its argument
    """
    return x

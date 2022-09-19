# Copyright (c) 2022 Philip May, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""LightGBM Metrics."""

from dataclasses import dataclass
from typing import Callable, List

import lightgbm as lgbm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class LightGbmEvalFunction:
    """Configuration of a LightGBM evaluation (metric) function.

    Args:
        name: The name of the metric.
        function: The metric function in scikit-learn metric style.
            First argument is y_true (1d array-like, or label indicator array / sparse matrix).
            Second argument is y_pred (1d array-like, or label indicator array / sparse matrix).
            Returns the calculated metric.
        is_higher_better: Indicates if higher metric result is better.
        needs_binary_predictions: If the metric functions needs binary predictions (0 or 1) or
            the raw logistic output.
    """

    name: str
    function: Callable
    is_higher_better: bool
    needs_binary_predictions: bool


def binary_eval_callback_factory(lightgbm_eval_functions: List[LightGbmEvalFunction]):
    """Factory function for a binary evaluation callback for LightGBM.

    This functions needs a list of ``LightGbmEvalFunction``.
    From this list it constructs a callback function for LightGBM.
    This callback function can be assigned to the ``feval`` parameter in
    ``lightgbm.train``.
    """

    def binary_eval_callback(y_pred: np.ndarray, data: lgbm.basic.Dataset):
        assert y_pred.ndim == 1
        y_true = data.get_label()
        y_pred_binary = None  # we do lasy init here (see below)
        results = []
        for lightgbm_eval_function in lightgbm_eval_functions:
            if lightgbm_eval_function.needs_binary_predictions:
                if y_pred_binary is None:  # we do lasy init here
                    y_pred_binary = np.round(y_pred)
                result = lightgbm_eval_function.function(y_true, y_pred_binary)
            else:
                result = lightgbm_eval_function.function(y_true, y_pred)
            results.append(
                (lightgbm_eval_function.name, result, lightgbm_eval_function.is_higher_better)
            )
        return results

    return binary_eval_callback


lgbm_f1_score = LightGbmEvalFunction(
    name="f1",
    function=f1_score,
    is_higher_better=True,
    needs_binary_predictions=True,
)

lgbm_accuracy_score = LightGbmEvalFunction(
    name="accuracy",
    function=accuracy_score,
    is_higher_better=True,
    needs_binary_predictions=True,
)

lgbm_average_precision_score = LightGbmEvalFunction(
    name="average_precision",
    function=average_precision_score,
    is_higher_better=True,
    needs_binary_predictions=False,
)

lgbm_roc_auc_score = LightGbmEvalFunction(
    name="roc_auc",
    function=roc_auc_score,
    is_higher_better=True,
    needs_binary_predictions=False,
)


lgbm_recall_score = LightGbmEvalFunction(
    name="recall",
    function=recall_score,
    is_higher_better=True,
    needs_binary_predictions=True,
)


lgbm_precision_score = LightGbmEvalFunction(
    name="precision",
    function=precision_score,
    is_higher_better=True,
    needs_binary_predictions=True,
)

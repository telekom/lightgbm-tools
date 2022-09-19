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
    """TODO: add docstring."""

    name: str
    function: Callable
    is_higher_better: bool
    needs_binary_predictions: bool


def binary_eval_callback_factory(lightgbm_eval_functions: List[LightGbmEvalFunction]):
    """TODO: add docstring."""

    def binary_eval_callback(y_pred: np.ndarray, data: lgbm.basic.Dataset):
        """TODO: add docstring."""
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


f1_score = LightGbmEvalFunction(
    name="f1",
    function=f1_score,
    is_higher_better=True,
    needs_binary_predictions=True,
)

accuracy_score = LightGbmEvalFunction(
    name="accuracy",
    function=accuracy_score,
    is_higher_better=True,
    needs_binary_predictions=True,
)

average_precision_score = LightGbmEvalFunction(
    name="average_precision",
    function=average_precision_score,
    is_higher_better=True,
    needs_binary_predictions=False,
)

roc_auc_score = LightGbmEvalFunction(
    name="roc_auc",
    function=roc_auc_score,
    is_higher_better=True,
    needs_binary_predictions=False,
)


recall_score = LightGbmEvalFunction(
    name="recall",
    function=recall_score,
    is_higher_better=True,
    needs_binary_predictions=True,
)


precision_score = LightGbmEvalFunction(
    name="precision",
    function=precision_score,
    is_higher_better=True,
    needs_binary_predictions=True,
)

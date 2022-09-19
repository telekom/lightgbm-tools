# Copyright (c) 2022 Philip May, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""LightGBM Metrics."""

from dataclasses import dataclass
from typing import Callable, List

import lightgbm as lgbm
import numpy as np


# from sklearn.metrics import accuracy_score, average_precision_score, f1_score


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
        y_pred_binary = np.round(y_pred)
        results = []
        for lightgbm_eval_function in lightgbm_eval_functions:
            if lightgbm_eval_function.needs_binary_predictions:
                result = lightgbm_eval_function.function(y_true, y_pred_binary)
            else:
                result = lightgbm_eval_function.function(y_true, y_pred)
            results.append(
                (lightgbm_eval_function.name, result, lightgbm_eval_function.is_higher_better)
            )
        return results

    return binary_eval_callback

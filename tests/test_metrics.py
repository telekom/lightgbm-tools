# Copyright (c) 2022 Philip May, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from math import isclose

import lightgbm as lgbm
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score

from lightgbm_tools.metrics import (
    binary_eval_callback_factory,
    lgbm_accuracy_score,
    lgbm_average_precision_score,
    lgbm_f1_score,
)


def test_binary_eval_callback_factory_accuracy():
    y_pred = np.array([0.2, 0.3, 0.3, 0.9])
    label = [0, 0, 1, 1]
    data = lgbm.Dataset(None, label=label)
    accuracy_callback = binary_eval_callback_factory([lgbm_accuracy_score])
    result = accuracy_callback(y_pred, data)
    print(result)

    assert result is not None
    assert len(result) == 1
    assert len(result[0]) == 3
    assert result[0][0] == "accuracy"
    assert isclose(result[0][1], accuracy_score(label, np.round(y_pred)))
    assert result[0][2]


def test_binary_eval_callback_factory_f1():
    y_pred = np.array([0.2, 0.3, 0.3, 0.9])
    label = [0, 0, 1, 1]
    data = lgbm.Dataset(None, label=label)
    accuracy_callback = binary_eval_callback_factory([lgbm_f1_score])
    result = accuracy_callback(y_pred, data)
    print(result)

    assert result is not None
    assert len(result) == 1
    assert len(result[0]) == 3
    assert result[0][0] == "f1"
    assert isclose(result[0][1], f1_score(label, np.round(y_pred)))
    assert result[0][2]


def test_binary_eval_callback_factory_average_precision():
    y_pred = np.array([0.2, 0.3, 0.3, 0.9])
    label = [0, 0, 1, 1]
    data = lgbm.Dataset(None, label=label)
    accuracy_callback = binary_eval_callback_factory([lgbm_average_precision_score])
    result = accuracy_callback(y_pred, data)
    print(result)

    assert result is not None
    assert len(result) == 1
    assert len(result[0]) == 3
    assert result[0][0] == "average_precision"
    assert isclose(result[0][1], average_precision_score(label, y_pred))
    assert result[0][2]

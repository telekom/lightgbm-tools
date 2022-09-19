# Copyright (c) 2022 Philip May, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Main example for usage."""

from pprint import pprint
from typing import Dict

import lightgbm as lgbm
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from lightgbm_tools.metrics import (
    LightGbmEvalFunction,
    binary_eval_callback_factory,
    lgbm_accuracy_score,
    lgbm_average_precision_score,
    lgbm_f1_score,
)


# load the breast cancer data
# see: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer  # noqa: E501
breast_cancer_data = load_breast_cancer()

# split data to dataset (x) and labels (y)
x = breast_cancer_data["data"]
y = breast_cancer_data["target"]

print("x.shape", x.shape)
print("y.shape", y.shape)

# split data to train and validation
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_val.shape:", x_val.shape)
print("y_val.shape:", y_val.shape)

# create own custom eval (metric) function for balanced_accuracy_score
lgbm_balanced_accuracy = LightGbmEvalFunction(
    name="balanced_accuracy",
    function=balanced_accuracy_score,
    is_higher_better=True,
    needs_binary_predictions=True,
)


# use the factory function to create the callback
# add the predefined F1, accuracy and average precision metrics
# and the own custom eval (metric) function for balanced_accuracy_score
callback = binary_eval_callback_factory(
    [lgbm_f1_score, lgbm_accuracy_score, lgbm_average_precision_score, lgbm_balanced_accuracy]
)

# create LightGBM datasets
train_data = lgbm.Dataset(x_train, label=y_train)
val_data = lgbm.Dataset(x_val, label=y_val)

param = {
    "objective": "binary",
}

evals_result: Dict = {}

bst = lgbm.train(
    param,
    train_data,
    valid_sets=val_data,
    num_boost_round=6,
    verbose_eval=False,
    evals_result=evals_result,
    feval=callback,  # here we pass the callback
)

pprint(evals_result)

# LightGBM Tools

[![MIT License](https://img.shields.io/github/license/telekom/lightgbm-tools)](https://github.com/telekom/lightgbm-tools/blob/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/lightgbm-tools)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/lightgbm-tools.svg)](https://pypi.python.org/pypi/lightgbm-tools)

This Python package implements tools for [LightGBM](https://lightgbm.readthedocs.io/).
In the current version lightgbm-tools focuses on binary classification metrics.

## What exact problem does this tool solve?

LightGBM has some [built-in metrics](https://lightgbm.readthedocs.io/en/v3.3.2/Parameters.html#metric) that can be used.
These are useful but limited. Some important metrics are missing.
These are, among others, the [F1-score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
and the [average precision (AP)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

These metrics can be easily added using this tool.
This happens through a mechanism built into LightGBM where we can assign a callback to the `feval` parameter of
`lightgbm.train` (see
[lightgbm.train](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html#lightgbm-train) documentation).

## Maintainers

[![One Conversation](https://raw.githubusercontent.com/telekom/lightgbm-tools/main/docs/source/imgs/1c-logo.png)](https://www.welove.ai/)
<br/>
This project is maintained by the [One Conversation](https://www.welove.ai/)
team of [Deutsche Telekom AG](https://www.telekom.com/).

## Usage

You can find a fully functional example here: <https://github.com/telekom/lightgbm-tools/blob/main/examples/main_usage.py>

The easiest way is to use the predefined callback functions. These are:

- `lightgbm_tools.metrics.lgbm_f1_score_callback`
- `lightgbm_tools.metrics.lgbm_accuracy_score_callback`
- `lightgbm_tools.metrics.lgbm_average_precision_score_callback`
- `lightgbm_tools.metrics.lgbm_roc_auc_score_callback`
- `lightgbm_tools.metrics.lgbm_recall_score_callback`
- `lightgbm_tools.metrics.lgbm_precision_score_callback`

Here F1 is used as an example to show how the predefined callback functions can be used:

```python
import lightgbm
from lightgbm_tools.metrics import lgbm_f1_score_callback

bst = lightgbm.train(
    params,
    train_data,
    valid_sets=val_data,
    num_boost_round=6,
    verbose_eval=False,
    evals_result=evals_result,
    feval=lgbm_f1_score_callback,  # here we pass the callback to LightGBM
)
```

You can also reuse other implementations of metrics.
Here is an example of how to do this using the `balanced_accuracy_score` from scikit-learn:

```python
import lightgbm
from sklearn.metrics import balanced_accuracy_score
from lightgbm_tools.metrics import LightGbmEvalFunction, binary_eval_callback_factory

# define own custom eval (metric) function for balanced_accuracy_score
lgbm_balanced_accuracy = LightGbmEvalFunction(
    name="balanced_accuracy",
    function=balanced_accuracy_score,
    is_higher_better=True,
    needs_binary_predictions=True,
)

# use the factory function to create the callback
lgbm_balanced_accuracy_callback = binary_eval_callback_factory([lgbm_balanced_accuracy])

bst = lightgbm.train(
    params,
    train_data,
    valid_sets=val_data,
    num_boost_round=6,
    verbose_eval=False,
    evals_result=evals_result,
    feval=lgbm_balanced_accuracy_callback,  # here we pass the callback to LightGBM
)
```

This tool can also be used to calculate multiple metrics at the same time.
It can be done by passing several definitions of metrics (in a list) to the
`binary_eval_callback_factory`.
The followring predefined metric definitions (`LightGbmEvalFunction`) are available:

- `lightgbm_tools.metrics.lgbm_f1_score`
- `lightgbm_tools.metrics.lgbm_accuracy_score`
- `lightgbm_tools.metrics.lgbm_average_precision_score`
- `lightgbm_tools.metrics.lgbm_roc_auc_score`
- `lightgbm_tools.metrics.lgbm_recall_score`
- `lightgbm_tools.metrics.lgbm_precision_score`

Below is an example how to combine F1 and average precision:

```python
import lightgbm
from lightgbm_tools.metrics import (
    binary_eval_callback_factory,
    lgbm_average_precision_score,
    lgbm_f1_score,
)

# use the factory function to create the callback
callback = binary_eval_callback_factory([lgbm_average_precision_score, lgbm_f1_score])

bst = lightgbm.train(
    params,
    train_data,
    valid_sets=val_data,
    num_boost_round=6,
    verbose_eval=False,
    evals_result=evals_result,
    feval=callback,  # here we pass the callback to LightGBM
)
```

## Installation

lightgbm-tools can be installed with pip:

```bash
pip install lightgbm-tools
```

To do development and run unit tests locally, ensure that you have installed all relevant requirements.
You will probably want to install it in "editable mode" if you are developing locally:

```bash
pip install -e .[all]
```

## Licensing

Copyright (c) 2022 [Philip May](https://may.la/), [Deutsche Telekom AG](https://www.telekom.com/)

Licensed under the **MIT License** (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License by reviewing the file
[LICENSE](https://github.com/telekom/lightgbm-tools/blob/main/LICENSE) in the repository.

# LightGBM Tools

## Usage

For a full example see here: <https://github.com/telekom/lightgbm-tools/blob/main/examples/main_usage.py>

Create own custom eval (metric) function:

```python
from sklearn.metrics import balanced_accuracy_score
from lightgbm_tools.metrics import LightGbmEvalFunction

# create own custom eval (metric) function for balanced_accuracy_score
lgbm_balanced_accuracy = LightGbmEvalFunction(
    name="balanced_accuracy",
    function=balanced_accuracy_score,
    is_higher_better=True,
    needs_binary_predictions=True,
)
```

Create the callback function for LightGBM:

```python
from lightgbm_tools.metrics import (
    binary_eval_callback_factory,
    lgbm_accuracy_score,
    lgbm_average_precision_score,
    lgbm_f1_score,
)

# use the factory function to create the callback
# add the predefined F1, accuracy and average precision metrics
# and the own custom eval (metric) function for balanced_accuracy_score
callback = binary_eval_callback_factory(
    [lgbm_f1_score, lgbm_accuracy_score, lgbm_average_precision_score, lgbm_balanced_accuracy]
)
```

Use the callback:

```python
import lightgbm as lgbm

bst = lgbm.train(
    param,
    train_data,
    valid_sets=val_data,
    num_boost_round=6,
    verbose_eval=False,
    evals_result=evals_result,
    feval=callback,  # here we pass the callback
)
```

## Licensing

Copyright (c) 2022 Philip May, Deutsche Telekom AG

Licensed under the **MIT License** (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License by reviewing the file
[LICENSE](https://github.com/telekom/lightgbm-tools/blob/main/LICENSE) in the repository.

# CMSC-727-project


# Usage

`python rpropcifar.py --epochs 2  --use_rprop False`

The rest of the parameters will be optimized by Optuna:

- Batch size
- Momentum
- Learning rate

The current version optimizes all 3 of those when rprop is False (so using SGD) and only optimizes batch size when rprop is True.
I will update this soon so it works fully for both, but for now we can optimize the baseline model with SGD not rprop.


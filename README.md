# CMSC-727-project


# Usage
`optional arguments:`

  `-h, --help            show this help message and exit`
  
  `--epochs EPOCHS       number of epochs for training`
 
  `--num_trials NUM_TRIALS
                        The number of times Optuna will train the model.
                        Higher means better optimization, but longer training
                        time`
 
  `--use_rprop USE_RPROP
                        True if using rprop, False if using sgd`
`

# Example use case
`python rpropcifar.py --epochs 2 --num_trials 10  --use_rprop False`

The rest of the parameters will be optimized by Optuna:

- Batch size
- Momentum
- Learning rate

The current version optimizes all 3 of those when rprop is False (so using SGD) and only optimizes batch size when rprop is True.
I will update this soon so it works fully for both, but for now we can optimize the baseline model with SGD not rprop.


Ideally, we should run for at least 100 trials with 30-100 epochs each, but we will need a GPU for this.

TODO:
- gpu
- use validation set for optimization and then test after optimization had yielded best parameters

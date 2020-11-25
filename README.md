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

# Example Use Case
`python rpropcifar.py --epochs 2 --num_trials 10  --use_rprop False`

This will train a model for 2 epochs using SGD optimizer. It will search the paramter space by retraining the model 10 times, and print the best parameters it found at the end.

# The Optimization
These parameters will be optimized by Optuna:

- Batch size
- Momentum
- Learning rate

Things we are not currently optimizing for but might want to:
- etas and stepsizes (for RPROP only. See documentation [https://pytorch.org/docs/stable/optim.html])
- sizes of layers in the network
- number of layers in the network
- activation function 
- type of pooling

The current version optimizes all 3 of those when rprop is False (so using SGD) and only optimizes batch size and learning rate when rprop is True.
I will update this soon so it works fully for both, but for now we can optimize the baseline model with SGD not rprop.

Ideally, we should run for at least 100 trials with 30-100 epochs each, but we will need a GPU for this.

# To Do
TODO:
- get the code running on a GPU
- run many trials with many epochs
- decide which parameters shoulde be optimized for each method (SGD and RPROP)
- use validation set for optimization and then test after optimization had yielded best parameters
- make some nice plots to show the data from the experiments

TODO Later (Maybe):
- compare against other optimizers
- another dataset
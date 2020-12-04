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
                        Use this flag if you want to use rprop. --use_rprop True will make rprop the optimizer`
`

# Example Use Case
First, run something like:
`python rpropcifar.py --epochs 50 --num_trials 50`

This will train a model for 50 epochs using SGD optimizer. It will search the parameter space by retraining the model 50 times, and print the best parameters it found at the end.
The output will look like (just an example, the numbers will likely be different):

`The best parameters are:`

`{'learning_rate': 0.006510474124597193, 'momentum': 0.3450199439278914, 'batch_size': 8}`

You can then get the test accuracy by running the model with those parameters. To do this you must provide the parameters via command line and specify the number of trials to be 0, since we are no longer optimizing. An example:

`python rpropcifar.py --epochs 50  --use_rprop False --num_trials 0 --learning_rate 0.01 --momentum 0.35 --batch_size 8`

This will produce an overall test accuracy, as well as a class by class accuracy.

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
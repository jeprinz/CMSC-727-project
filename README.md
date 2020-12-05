# CMSC-727-project


# Usage
```
optional arguments:

  -h, --help            show this help message and exit
 
  --epochs EPOCHS       number of epochs for training
 
  --num_trials NUM_TRIALS
                        The number of times Optuna will train the model.
                        Higher means better optimization, but longer training
                        time
 
  --batch_size [BATCH_SIZE]
                        number of samples per training batch
 
  --learning_rate [LEARNING_RATE]
                        the learning rate between 0 and 1
 
  --momentum [MOMENTUM]
                        the momentum to use betweeon 0 and 1
 
  --eta_minus [ETA_MINUS]
                        the eta lower bound for RPROP only
 
  --eta_plus [ETA_PLUS]
                        the eta upper bound for RPROP only
 
  --step_minus [STEP_MINUS]
                        the step size lower bound for RPROP only
 
  --step_plus [STEP_PLUS]
                        the step size upper bound for RPROP only
 
  --num_filters [NUM_FILTERS]
                        how big the conv layer should be
 
  --fc1_size [FC1_SIZE]
                        how big the first fully connected layer should be
 
  --fc2_size [FC2_SIZE]
                        how big the second fully connected layer should be
 
  --use_rprop USE_RPROP
                        True if using rprop, False if using sgd
 
  --save_weights SAVE_WEIGHTS
                        True if saving weights to pickle file
```

# Example Use Case
First, run something like:
`python rpropcifar.py --epochs 50 --num_trials 50`

This will train a model for 50 epochs using SGD optimizer. It will search the parameter space by retraining the model 50 times, and print the best parameters it found at the end.
The output will look like (just an example, the numbers will likely be different):

`The best parameters are:`

`{'learning_rate': 0.12776731038932096, 'batch_size':
4, 'num_filters': 7.0, 'fc1_size': 135.0, 'fc2_size': 84.0, 'eta_minus': 0.35346034211596966, 'eta_plus': 1.3655544555547428, 'step_m
inus': 1.147102403767491e-06, 'step_plus': 48.52966612009824}`

You can then get the test accuracy by running the model with those parameters. To do this you must provide the parameters via command line and specify the number of trials to be 0, since we are no longer optimizing. An example:

`python rpropcifar.py --epochs 50  --use_rprop False --num_trials 0 --learning_rate 0.01 --batch_size 8`

This will produce an overall test accuracy, as well as a class by class accuracy.

# The Optimization
These parameters will be optimized by Optuna:

- Batch size
- Learning rate
- Number of filters in convolution layer 2
- size of hidden layer 1
- size of hidden layer 2
- For RPROP only:
    - Etas
    - Stepsizes

Things we are not currently optimizing for but might want to in future work:
- number of layers in the network
- activation function
- type of pooling
- amount of dropout
from tensorflow.keras.backend import clear_session
import gc

clear_session()

gc.collect()

from model_manager import section1, section2, single_model
import optimizer as opt
import dataset_creator
from constants import *

optim = opt.BayesianOptimizer()
params = {
    'max_conv_layers': 5,
    'max_dense_layers': 5,
    'max_conv_size': 256,
    'max_dense_size': 256,
    'epochs': 1
}
model = section1()
best_trial = optim.optimize(model, target=3, initial_params=params, max_trials=10, write_to_file=True)
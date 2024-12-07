from tensorflow.keras.backend import clear_session
import gc

clear_session()

gc.collect()

from model_manager import section1, section2, single_model
import optimizer as opt
import dataset_creator

best_conv = [64, 16, 3] 
best_dense = [16, 32, 64]

checkpoints_sect1 = './checkpoints_sect1/'
dataset2_gen = dataset_creator.Dataset2(weights=checkpoints_sect1 + 'model_optimize_3.weights.h5', conv_layers=best_conv, dense_layers=best_dense)
dataset2_gen.generate_dataset()
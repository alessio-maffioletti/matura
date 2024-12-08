from tensorflow.keras.backend import clear_session
import gc

clear_session()

gc.collect()


from model_manager import section1, section2, single_model
import optimizer as opt
import dataset_creator
from constants import *

dataset1_generator = dataset_creator.Dataset1()
dataset1_generator.generate_dataset(train_size=10000, test_size=10000, whole_dataset=True)

conv_layers = [128,32,8]
dense_layers = [32,64,128]
target = 3
model = section1()
optimizer = opt.Optimizer()
best_conv, best_dense, iter_num = optimizer.optimize(model, conv_layers, dense_layers, target, max_iter=3, max_failed_trains=2)
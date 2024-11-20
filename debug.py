from tensorflow.keras.backend import clear_session
import gc

clear_session()

gc.collect()


from train_models_tfrecord import single, sect1, sect2

net3 = sect2()
net3.initialise_data_and_model()
params = {'epochs': 20,
        'tensorboard': True, 
        'cp_callback': True,
        'weights': None
        }
net3.train(params=params)

net3.train_debug2()


net1 = sect1()
net1.initialise_data_and_model()
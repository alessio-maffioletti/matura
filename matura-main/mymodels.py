import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard, Callback
from tensorflow.keras import optimizers
import subprocess
import random
from constants import *
import time
import sys

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

powershell_executable = 'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe'
powershell_command = 'Get-ChildItem -Path "H:\\aless\\Documents\\Python_Scripts\\Matur\\matura-private-main\\logs" | Remove-Item -Recurse -Force'



# alle custom Callbacks inklusiv wrapper und print_trainable_params wurden mit ChatGPT generiert, aber ausführlich geprüft und verstanden
class SingleLineProgressBar(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        progress_message = f"\rEpoch {epoch + 1} / {self.params['epochs']} - "

        for key, value in logs.items():
            progress_message += f"{key}: {value:.4f} - "

        sys.stdout.write(progress_message)
        sys.stdout.flush()

    def on_train_end(self, logs=None):
        print()

class StopAtAccuracy(Callback):
    def __init__(self, target_accuracy):
        super().__init__()
        self.target_accuracy = target_accuracy
        self.reached_target = False

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get("val_accuracy")
        if accuracy is not None and accuracy >= self.target_accuracy:
            print(f"\nTarget accuracy of {self.target_accuracy} reached! Stopping training.")
            self.model.stop_training = True
            self.reached_target = True

class StopAtMAE(Callback):
    def __init__(self, target_mae):
        super().__init__()
        self.target_mae = target_mae
        self.reached_target = False

    def on_epoch_end(self, epoch, logs=None):
        mae = logs.get("val_mean_absolute_error")
        if mae is not None and mae <= self.target_mae:
            print(f"\nTarget MAE of {self.target_mae} reached! Stopping training.")
            self.model.stop_training = True
            self.reached_target = True

class StopAtTime(Callback):
    def __init__(self, time_limit):
        super().__init__()
        self.time_limit = time_limit
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.time_limit:
            print(f"\nTime limit of {self.time_limit} seconds exceeded. Stopping training.")
            self.model.stop_training = True


class BatchLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.history = {}
        self.batch = []

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.batch.append(len(self.batch))

        for metric, value in logs.items():
            if metric not in self.history:
                self.history[metric] = []
            self.history[metric].append(value)

    def on_test_batch_end(self, batch, logs=None):
        logs = logs or {}

        for metric, value in logs.items():
            val_metric = f"val_{metric}"
            if val_metric not in self.history:
                self.history[val_metric] = []
            self.history[val_metric].append(value)

class BatchRunWrapper:
    def __init__(self, batch_logger):
        self.history = batch_logger.history
        self.epoch = batch_logger.batch 



def print_trainable_params(model, figsize=(2, 2), threshold=5):
    total_trainable_params = 0
    layer_names = []
    layer_params = []
    layer_colors = []

    dense_color = 'red'  
    conv_color = 'blue'  

    for idx, layer in enumerate(model.layers):
        if layer.trainable and len(layer.trainable_weights) > 0: 
            num_params = np.prod(layer.trainable_weights[0].shape)  
            total_trainable_params += num_params
            layer_names.append(layer.name)
            layer_params.append(num_params)

            if isinstance(layer, tf.keras.layers.Dense):
                color = dense_color
            elif isinstance(layer, tf.keras.layers.Conv2D):
                color = conv_color
            else:
                color = 'gray'  

            layer_colors.append(color)

    fig, ax = plt.subplots(figsize=figsize)  
    
    def percentage_label(x):
        return f"{x:.0f}%" if x > threshold else ""  

    wedges, texts, autotexts = ax.pie(layer_params, autopct=lambda p: percentage_label(p), startangle=140,
                                      colors=layer_colors, wedgeprops={'edgecolor': 'black'})  
    ax.set_title(f"Total: {total_trainable_params:,} params")
    ax.axis('equal')  

    plt.show()

    total_trainable_params = int(total_trainable_params)

    return total_trainable_params

def _set_initial_params(initial_params):
        default_params = {
            'flatten_type': 'flatten',
            'conv_layers': [32, 64],
            'dense_layers': [128, 64],
            'activation': 'relu',
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'dropout': 0.01
            }

        if initial_params is None:
            initial_params = default_params
        else:
            for key, value in default_params.items():
                initial_params.setdefault(key, value)

        return initial_params


# alle diese Modelle nutzen eine Struktur, welche aus Codebeispielen von einem Machine-Learning Kurs aus der ETH. Die Codes vom Workshop sind in der MA angehängt

class ClassificationModel:
    def __init__(self, input_shape, output_shape, activation, trainable_params=None):
        self.name = "model"
        
        self.trainable_params = _set_initial_params(trainable_params)

        initializer = tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED)
        
        self.model = models.Sequential()


        self.model.add(layers.Input(input_shape))

        for conv_layer in self.trainable_params['conv_layers']:
            self.model.add(layers.Conv2D(conv_layer, (3, 3), activation=self.trainable_params['activation'], kernel_initializer=initializer))
            #self.model.add(layers.BatchNormalization())
            self.model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='valid'))

        if self.trainable_params['flatten_type'] == 'flatten':
            self.model.add(layers.Flatten())
        elif self.trainable_params['flatten_type'] == 'global_average':
            self.model.add(layers.GlobalAveragePooling2D())

        for dense_layer in self.trainable_params['dense_layers']:
            self.model.add(layers.Dense(dense_layer, activation=self.trainable_params['activation'], kernel_initializer=initializer))
            self.model.add(layers.Dropout(self.trainable_params['dropout'], seed=RANDOM_SEED))

        self.model.add(layers.Dense(output_shape, activation=activation) )

        #print(self.model.summary())
    
    def compile(self, optimizer, loss, metrics):
        if self.trainable_params['optimizer'] == 'adam':
            optimizer = optimizers.Adam(learning_rate=self.trainable_params['learning_rate'])
        elif self.trainable_params['optimizer'] == 'sgd':
            optimizer = optimizers.SGD(learning_rate=self.trainable_params['learning_rate'], clipnorm=1.0)
        else:
            raise ValueError("Invalid optimizer name or learning rate, please choose between adam or sgd")
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(f"model compiled with params: {self.trainable_params}")
        trainable_params = print_trainable_params(self.model)
        return trainable_params
    
    def train(self, train_dataset, val_dataset, metric, params, checkpoints_folder=None):
        default_params = {
            'epochs': 10,
            'tensorboard': False,
            'cp_callback': False,
            'save_final': False,
            'weights': None,
            'stop_at': None,
            'weight_string': 'final',
            'max_time': None,
            'show_progress': True,
            'strop': True, 
            'batch_plot': True
            }
        
        if not params:
            params = default_params
        else:
            for key, value in default_params.items():
                params.setdefault(key, value)

        
        callbacks = []
        if params['show_progress']:
            callbacks.append(SingleLineProgressBar())
        
        if params['strop']:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3))
        
        
        if params['tensorboard']:
            if not params['weights']:
                subprocess.run(
                    [powershell_executable, '-Command', powershell_command],
                    stdout=subprocess.DEVNULL,  
                    stderr=subprocess.DEVNULL   
                )
            
            tensorboard_callback = TensorBoard(log_dir=LOGS_FOLDER)
            callbacks.append(tensorboard_callback)

        if params['cp_callback']:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoints_folder, f"{self.name}{params['weight_string']}{{epoch:02d}}.weights.h5"),
                save_weights_only=True,
                verbose=0
            )
            callbacks.append(cp_callback)

        if params['weights']:
            file_path = checkpoints_folder + f"/{self.name}{params['weight_string']}.weights.h5"
            self.model.load_weights(file_path)

        if params['stop_at']:
            if metric == 'mae':
                early_stopping = StopAtMAE(params['stop_at'])
                callbacks.append(early_stopping)
            elif metric == 'accuracy':
                early_stopping = StopAtAccuracy(params['stop_at'])
                callbacks.append(early_stopping)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        else:
            early_stopping = None

        if params['max_time']:
            callbacks.append(StopAtTime(params['max_time']))

        if params['batch_plot']:
            batch_log_callback = BatchLoggingCallback()
            callbacks.append(batch_log_callback)

        model_run = self.model.fit(
            train_dataset,
            epochs=params['epochs'],
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=0
        )

        if params['save_final']:
            self.model.save_weights(os.path.join(checkpoints_folder, f"{self.name}{params['weight_string']}.weights.h5"))

        if params['batch_plot']:
            model_run = BatchRunWrapper(batch_log_callback)

        return model_run, early_stopping.reached_target if early_stopping else False
    

    def load_weights(self, path):
        self.model.load_weights(path)
    
    def evaluate(self, dataset, weight_path=None):
        if weight_path is not None and os.path.exists(weight_path):
            self.model.load_weights(weight_path)

        predictions = self.model.predict(dataset, verbose=1)

        return predictions 
    
    def predict(self, input):
        return self.model.predict(input, verbose=0)

class RegressionModel(ClassificationModel):
    def __init__(self, input_shape, output_shape, activation, **kwargs):
        super().__init__(input_shape, output_shape, activation, **kwargs)
        self.name = "regression_model"
    
    def train(self, **kwargs):
        if not 'metric' in kwargs:
            kwargs['metric'] = 'mae'
        return super().train(**kwargs)
    
    def compile(self, optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error']):
        return super().compile(optimizer, loss, metrics)
    


class SingleModel(ClassificationModel):
    def __init__(self, trainable_params):
        self.name = "single_model"

        self.trainable_params = _set_initial_params(trainable_params)

        initializer = tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED)

        image_input = layers.Input(shape=INPUT_SHAPE, name='image')
        x = image_input

        for conv_layer in self.trainable_params['conv_layers']:
            x = layers.Conv2D(conv_layer, kernel_size=(3, 3), activation='relu', kernel_initializer=initializer)(x)
            x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

        #x = layers.Flatten()(x)
        x = layers.GlobalAveragePooling2D()(x)

        coordinates_input = layers.Input(shape=COORDS_SHAPE, name='coords')
        y = coordinates_input
        y = layers.Dense(64, activation='relu')(y)  
        combined = layers.Concatenate()([x, y])

        z = combined
        for dense_layer in self.trainable_params['dense_layers']:
            z = layers.Dense(dense_layer, activation='relu', kernel_initializer=initializer)(z)
            z = layers.Dropout(self.trainable_params['dropout'], seed=RANDOM_SEED)(z)  

        output = layers.Dense(LABELS_OUTPUT_SHAPE, activation=CLASSIFICATION_ACTIVATION, name='label')(z)

        self.model = models.Model(inputs=[image_input, coordinates_input], outputs=output)

        #print(self.model.summary())

    
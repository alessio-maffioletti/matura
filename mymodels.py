import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard
import subprocess
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

powershell_executable = 'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe'
powershell_command = 'Get-ChildItem -Path "H:\\aless\\Documents\\Python_Scripts\\Matur\\matura-private-main\\logs" | Remove-Item -Recurse -Force'

class SingleLineProgressBar(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        # Start with epoch information
        progress_message = f"\rEpoch {epoch + 1}/{self.params['epochs']} - "
        
        # Add all available metrics dynamically
        metrics = [f"{key}: {logs[key]:.4f}" for key in logs.keys() if key != 'batch']
        progress_message += " - ".join(metrics)
        
        print(progress_message, end='')  # Print on the same line

    def on_train_end(self, logs=None):
        print()  # Move to the next line after training ends

class debug_model():
    def __init__(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(42, 42, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10, activation='softmax'))
        self.model.summary()
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    def train(self, train_dataset, val_dataset):
        model_run = self.model.fit(
            train_dataset,
            epochs=20,
            validation_data=val_dataset,
            verbose=1
        )
        return model_run
class sect1():
    def __init__(self, input_shape=(128, 128, 1)):
        self.name = "sect1"
        # Create the model
        self.model = models.Sequential()
        # Add convolutional layers
        self.model.add(layers.Input(input_shape))

        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Flatten the output of the last convolutional layer
        self.model.add(layers.Flatten())

        # Add fully connected (dense) layers
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dropout(0.5))

        # Output layer for regression to predict x and y coordinates
        self.model.add(layers.Dense(2))  # Output layer with 2 neurons (x and y)
    
    def compile(self):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=100000,
            decay_rate=1)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
        print(self.model.summary())

    def load_weights(self, path):
        self.model.load_weights(path)



    def train(self, train_dataset, val_dataset, params, logs_folder=None, checkpoints_folder=None):
        """
        Train the model using provided datasets and parameters.

        Parameters:
            train_dataset (tf.data.Dataset): Training dataset.
            val_dataset (tf.data.Dataset): Validation dataset.
            params (dict): Training parameters (epochs, batch size, etc.).
            logs_folder (str): Folder path for saving TensorBoard logs.
            checkpoints_folder (str): Folder path for saving model checkpoints.

        Returns:
            model_run (History): Training history object.
        """
        # Default parameters
        default_params = {
            'epochs': 10,
            'batch_size': 512,
            'tensorboard': True,
            'cp_callback': True
        }
        
        # Merge provided params with defaults
        if not params:
            params = default_params
        else:
            for key, value in default_params.items():
                params.setdefault(key, value)

        # Initialize callbacks
        callbacks = [SingleLineProgressBar()]
        if params['tensorboard']:
            if not params['weights']:
                subprocess.run(
                    [powershell_executable, '-Command', powershell_command],
                    stdout=subprocess.DEVNULL,  # Suppress standard output
                    stderr=subprocess.DEVNULL   # Suppress error output
                )
            
            tensorboard_callback = TensorBoard(log_dir=logs_folder)
            callbacks.append(tensorboard_callback)

        if params['cp_callback']:
            # Create a callback for saving model weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoints_folder, f"{self.name}_epoch_{{epoch:02d}}.weights.h5"),
                save_weights_only=True,
                verbose=0
            )
            callbacks.append(cp_callback)

        if params['weights']:
            file_path = checkpoints_folder + f"/{self.name}_epoch_{params['weights']}.weights.h5"
            self.model.load_weights(file_path)

        # Train the model
        model_run = self.model.fit(
            train_dataset,
            epochs=params['epochs'],
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=0
        )
        return model_run
    
    def train_min(self, train_dataset, val_dataset):
        model_run = self.model.fit(
        train_dataset,
        epochs=20,
        validation_data=val_dataset,
        verbose=1
        )
        return model_run
    
    def evaluate(self, dataset, weight_path):
        if os.path.exists(weight_path):
            self.model.load_weights(weight_path)

        predictions = self.model.predict(dataset, verbose=1)

        return predictions 
    
    def predict(self, input):
        return self.model.predict(input, verbose=0)
    
class sect2(sect1):
    def __init__(self, input_shape=(42, 42, 1)):
        self.name = "sect2"
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(42, 42, 1)),  # Input layer for images
            tf.keras.layers.Flatten(),  # Flatten the image into a vector
            tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer
            tf.keras.layers.Dense(10, activation='softmax')  # Output layer for 10 classes
        ])

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.model.summary())

class single(sect1):
    def __init__(self, input_shape=(128, 128,1)):
        self.name = "single"

        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(128, 128, 1)),  # Input layer for images
            tf.keras.layers.Flatten(),  # Flatten the image into a vector
            tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer
            tf.keras.layers.Dense(10, activation='softmax')  # Output layer for 10 classes
        ])

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.model.summary())


class debug(sect1):
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(128, 128, 1)),  # Input layer for images
            tf.keras.layers.Flatten(),  # Flatten the image into a vector
            tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer
            tf.keras.layers.Dense(2)  # Output layer for 10 classes
        ])
    
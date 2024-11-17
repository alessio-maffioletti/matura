import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard

class SingleLineProgressBar(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        progress_message = (
            f"\rEpoch {epoch + 1}/{self.params['epochs']} - "
            f"loss: {logs['loss']:.4f} - val_loss: {logs.get('val_loss', 'N/A'):.4f}"
        )
        print(progress_message, end='')  # Print on the same line

    def on_train_end(self, logs=None):
        print()  # Move to the next line after training ends

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
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    def load_weights(self, path):
        self.model.load_weights(path)


    def train(self, train_dataset, val_dataset, params, logs_folder, checkpoints_folder):
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

        # Train the model
        model_run = self.model.fit(
            train_dataset,
            epochs=params['epochs'],
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=0
        )
        return model_run

    
    def evaluate(self, X, y, weight_path):
        if os.path.exists(weight_path):
            self.model.load_weights(weight_path)
            
        return self.model.evaluate(X, y)    
    
    def predict(self, input):
        return self.model.predict(input, verbose=0)
    
class sect2(sect1):
    def __init__(self, input_shape=(42, 42, 1)):
        self.name = "sect2"

        self.model = models.Sequential()
        self.model.add(layers.Input(input_shape))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.5))

        self.model.add(layers.Dense(10, activation='softmax'))

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


class single(sect1):
    def __init__(self, input_shape=(128, 128, 1)):
        self.name = "single"

        self.model = models.Sequential()
        self.model.add(layers.Input(input_shape))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.5))

        self.model.add(layers.Dense(10, activation='softmax'))

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
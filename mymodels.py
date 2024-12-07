import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard, Callback
import subprocess
import random
from constants import *

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
        mae = logs.get("val_mean_absolute_error")  # Change to "mean_absolute_error" if no validation is used
        if mae is not None and mae <= self.target_mae:  # Check if MAE is below the target
            print(f"\nTarget MAE of {self.target_mae} reached! Stopping training.")
            self.model.stop_training = True
            self.reached_target = True


def print_trainable_params(model, figsize=(2, 2), threshold=5):
    total_trainable_params = 0
    layer_names = []
    layer_params = []
    layer_colors = []

    # Define base colors for Dense and Conv layers
    dense_color = 'red'  # Red color for Dense layers
    conv_color = 'blue'  # Blue color for Conv2D layers

    # Collect data for the layers and their parameters
    for idx, layer in enumerate(model.layers):
        if layer.trainable and len(layer.trainable_weights) > 0:  # Ensure the layer has trainable weights
            num_params = np.prod(layer.trainable_weights[0].shape)  # Number of parameters in the layer
            total_trainable_params += num_params
            layer_names.append(layer.name)
            layer_params.append(num_params)

            # Assign colors based on layer type
            if isinstance(layer, tf.keras.layers.Dense):
                # Dense layer (Red)
                color = dense_color
            elif isinstance(layer, tf.keras.layers.Conv2D):
                # Conv layer (Blue)
                color = conv_color
            else:
                # Default color if it's neither Dense nor Conv
                color = 'gray'  # Gray for other layers

            layer_colors.append(color)

    # Create a pie chart with adjustable figsize
    fig, ax = plt.subplots(figsize=figsize)  # Create a figure with the specified figsize
    
    # Custom function to conditionally show percentages
    def percentage_label(x):
        return f"{x:.0f}%" if x > threshold else ""  # Only show percentage if greater than threshold

    # Create the pie chart with black edges
    wedges, texts, autotexts = ax.pie(layer_params, autopct=lambda p: percentage_label(p), startangle=140,
                                      colors=layer_colors, wedgeprops={'edgecolor': 'black'})  # Black line between slices
    ax.set_title(f"Total: {total_trainable_params:,} params")
    ax.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.

    # Show the plot
    plt.show()

    total_trainable_params = int(total_trainable_params)

    return total_trainable_params



class ClassificationModel:
    def __init__(self, input_shape, output_shape, activation, conv_layers=[32,64], dense_layers=[128,64]):
        self.name = "model"

        self.model = models.Sequential()


        self.model.add(layers.Input(input_shape))

        for conv_layer in conv_layers:
            self.model.add(layers.Conv2D(conv_layer, (3, 3), activation='relu'))
            self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Flatten())

        for dense_layer in dense_layers:
            self.model.add(layers.Dense(dense_layer, activation='relu'))
            self.model.add(layers.Dropout(0.01))

        self.model.add(layers.Dense(output_shape, activation=activation) ) 

        print(self.model.summary())
    
    def train(self, train_dataset, val_dataset, params, checkpoints_folder, metric='accuracy'):
        # Default parameters
        default_params = {
            'epochs': 10,
            'batch_size': 512,
            'tensorboard': True,
            'cp_callback': True,
            'save_final': True,
            'weights': None,
            'stop_at': None,
            'weight_string': 'final',
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
            
            tensorboard_callback = TensorBoard(log_dir=LOGS_FOLDER)
            callbacks.append(tensorboard_callback)

        if params['cp_callback']:
            # Create a callback for saving model weights
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

        # Train the model
        model_run = self.model.fit(
            train_dataset,
            epochs=params['epochs'],
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=0
        )

        if params['save_final']:
            self.model.save_weights(os.path.join(checkpoints_folder, f"{self.name}{params['weight_string']}.weights.h5"))

        return model_run, early_stopping.reached_target if early_stopping else False
    
    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        trainable_params = print_trainable_params(self.model)
        return trainable_params

    def load_weights(self, path):
        self.model.load_weights(path)
    
    def evaluate(self, dataset, weight_path):
        if os.path.exists(weight_path):
            self.model.load_weights(weight_path)

        predictions = self.model.predict(dataset, verbose=1)

        return predictions 
    
    def predict(self, input):
        return self.model.predict(input, verbose=0)

class RegressionModel(ClassificationModel):
    def train(self, train_dataset, val_dataset, params,checkpoints_folder, metric='mae'):
        return super().train(train_dataset, val_dataset, params,checkpoints_folder, metric)
    
    def compile(self, optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error']):
        return super().compile(optimizer, loss, metrics)
    

class SingleModel(ClassificationModel):
    def __init__(self, conv_layers=[32,64], dense_layers=[128,64]):
        self.name = "single_model"
        image_input = layers.Input(shape=INPUT_SHAPE, name='image')  # Example shape for Conv3D
        x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(image_input)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Flatten()(x)

        # Coordinates Input
        coordinates_input = layers.Input(shape=COORDS_OUTPUT_SHAPE, name='coords')  # 2 for x and y
        y = layers.Dense(64, activation='relu')(coordinates_input)

        # Merge the two branches
        combined = layers.Concatenate()([x, y])
        z = layers.Dense(128, activation='relu')(combined)
        z = layers.Dense(64, activation='relu')(z)
        output = layers.Dense(LABELS_OUTPUT_SHAPE, activation=CLASSIFICATION_ACTIVATION, name='label')(z)  # Single number output

        # Define the model
        self.model = models.Model(inputs=[image_input, coordinates_input], outputs=output)

    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models

class sect1:
    def __init__(self, input_shape=(128, 128, 1)):
        # Create the model
        self.model = models.Sequential()

        # Add convolutional layers
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
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

    def load_weights(self, path=r'H:\aless\Documents\Python_Scripts\Matur\mnist\training_1\sect\model_epoch_09.weights.h5'):
        self.model.load_weights(path)
    
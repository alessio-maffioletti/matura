import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mymodels
import tensorflow as tf
from tensorflow.keras.backend import clear_session
import time
from constants import *

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
clear_session()


def _parse_image_function(example_proto, image_shape=[128,128,1], label_shape=[10]):
    # Define the features to be extracted (serialized image and label)
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),  # Expecting the image as a serialized tensor (string)
        'label': tf.io.FixedLenFeature([], tf.string),  # Expecting the label as a serialized tensor (string)
    }

    # Parse the input tf.train.Example proto using the dictionary
    parsed_features = tf.io.parse_single_example(example_proto, image_feature_description)
    
    # Deserialize the image and label tensors
    image = tf.io.parse_tensor(parsed_features['image'], out_type=tf.float32)  # Deserialize image tensor
    label = tf.io.parse_tensor(parsed_features['label'], out_type=tf.int32)  # Deserialize label tensor

    image_shape = [BATCH_SIZE, *image_shape]
    label_shape = [BATCH_SIZE, *label_shape]

    #print(image_shape, label_shape)

    # Ensure that the image tensor has the correct shape
    image.set_shape(image_shape)  # Set the known shape for the image tensor

    # Ensure that the label tensor has the correct shape
    label.set_shape(label_shape)

    return image, label

def _parse_image_function_2(example_proto, 
                            image_shape=[128, 128, 1], 
                            coords_shape=[2], 
                            label_shape=[10]):
    # Define the features to be extracted (serialized tensors for image, coords, and label)
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),   # Serialized image tensor
        'coords': tf.io.FixedLenFeature([], tf.string),  # Serialized coords tensor
        'label': tf.io.FixedLenFeature([], tf.string),   # Serialized label tensor
    }

    # Parse the input `tf.train.Example` proto using the dictionary
    parsed_features = tf.io.parse_single_example(example_proto, image_feature_description)
    
    # Deserialize the tensors
    image = tf.io.parse_tensor(parsed_features['image'], out_type=tf.float32)  # Deserialize image tensor
    coords = tf.io.parse_tensor(parsed_features['coords'], out_type=tf.float32)  # Deserialize coords tensor
    label = tf.io.parse_tensor(parsed_features['label'], out_type=tf.int32)  # Deserialize label tensor

    image_shape = [BATCH_SIZE, *image_shape]
    coords_shape = [BATCH_SIZE, *coords_shape]
    label_shape = [BATCH_SIZE, *label_shape]

    #print(image_shape, coords_shape, label_shape)
    # Set the correct shapes for the tensors
    image.set_shape(image_shape)    # Set the known shape for the image tensor
    coords.set_shape(coords_shape)  # Set the known shape for the coords tensor
    label.set_shape(label_shape)    # Set the known shape for the label tensor

    return (image, coords), label


class better_models:
    def initialise_data_and_model(self, train_dataset_path, val_dataset_path, image_shape, label_shape):
        clear_session()
        #'coords.tfrecord'
        #'coords_test.tfrecord'
        self.train_dataset = tf.data.TFRecordDataset(train_dataset_path).map(lambda example_proto: _parse_image_function(example_proto, image_shape=image_shape, label_shape=label_shape))
        self.val_dataset = tf.data.TFRecordDataset(val_dataset_path).map(lambda example_proto: _parse_image_function(example_proto, image_shape=image_shape, label_shape=label_shape))

    def train(self, checkpoints_folder, params):
        start_time = time.time()
        
        self.run, reached_target = self.model.train(self.train_dataset, self.val_dataset, params, checkpoints_folder)
        
        training_time = time.time() - start_time
        
        return reached_target, round(training_time, 1)
    def _plot_random(self, image, predicted, actual):
        plt.imshow(image, cmap='gray')
        #plt.scatter(predicted[0], predicted[1], color='red', label='Predicted')
        #plt.scatter(actual[0], actual[1], color='blue', label='Actual')
        #plt.legend()
        plt.title(f"Predicted: {predicted.argmax().round(1)} | Actual: {actual.argmax()}")
        plt.show()

    def eval_random(self):
        for a in self.val_dataset:  # Iterate through the dataset
            image_batch, label_batch = a  # Get a batch of images and labels
            image_batch = image_batch.numpy()  # Convert to numpy for easier handling
            label_batch = label_batch.numpy()
            random_sample = np.random.randint(0, image_batch.shape[0])  # Randomly select an image from the batch

            # Extract the selected image and label
            image = image_batch[random_sample]
            actual = label_batch[random_sample]

            # Make predictions
            predictions = self.model.predict(image_batch)
            predicted = predictions[random_sample]

            self._plot_random(image, predicted, actual)

            break  # Only evaluate one random sample

    def plot(self):

        history_model = self.run.history
        #print("The history has the following data: ", history_model.keys())

        fig, axs = plt.subplots(1, 2, figsize=(10, 2))

        # Plotting the training and validation accuracy during the training
        sns.lineplot(
            x=self.run.epoch, y=history_model[list(history_model)[0]], color="blue", label="Training set", ax=axs[0]
        )
        sns.lineplot(
            x=self.run.epoch,
            y=history_model[list(history_model)[2]],
            color="red",
            label="Valdation set",
            ax=axs[0],
        )
        axs[0].set_xlabel("epochs")
        axs[0].set_ylabel(list(history_model)[0])

        # Plotting the training and validation loss during the training
        sns.lineplot(
            x=self.run.epoch, y=history_model[list(history_model)[1]], color="blue", label="Training set", ax=axs[1]
        )
        sns.lineplot(
            x=self.run.epoch,
            y=history_model[list(history_model)[3]],
            color="red",
            label="Valdation set",
            ax=axs[1],
        )
        axs[1].set_xlabel("epochs")
        axs[1].set_ylabel(list(history_model)[1])

        plt.show()

class section1(better_models):
    def initialise_data_and_model(self, conv_layers=[32, 64], dense_layers=[128, 64]):
        
        super().initialise_data_and_model(train_dataset_path=TRAIN_COORDS_PATH, val_dataset_path=TEST_COORDS_PATH, image_shape=IMAGE_SHAPE, label_shape=COORDS_SHAPE)
    
        self.model = mymodels.RegressionModel(conv_layers=conv_layers, dense_layers=dense_layers, input_shape=INPUT_SHAPE, output_shape=COORDS_OUTPUT_SHAPE, activation=REGRESSION_ACTIVATION)
        trainable_params = self.model.compile()
        return trainable_params
        
    def train(self, params=None):
        return super().train(SECT1_CHECKPOINT_FOLDER, params)
    
    def _plot_random(self, image, predicted, actual):
        plt.imshow(image, cmap='gray')
        plt.scatter(predicted[0], predicted[1], color='red', label='Predicted')
        plt.scatter(actual[0], actual[1], color='blue', label='Actual')
        plt.legend()
        plt.title(f"Predicted: {predicted.round(1)} | Actual: {actual}")
        plt.show()
    
class section2(better_models):
    def initialise_data_and_model(self, conv_layers=[32, 64], dense_layers=[128, 64]):
        
        super().initialise_data_and_model(train_dataset_path=TRAIN_CROPPED_PATH, val_dataset_path=TEST_CROPPED_PATH, image_shape=CROPPED_IMAGE_SHAPE, label_shape=LABELS_SHAPE)

        self.model = mymodels.ClassificationModel(conv_layers=conv_layers, dense_layers=dense_layers, input_shape=CROPPED_INPUT_SHAPE, output_shape=LABELS_OUTPUT_SHAPE, activation=CLASSIFICATION_ACTIVATION)
        trainable_params = self.model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'] )
    
        return trainable_params
    
    def train(self, params=None):
        return super().train(SECT2_CHECKPOINT_FOLDER, params)
    
class single_model(better_models):
    def initialise_data_and_model(self, conv_layers=[32, 64], dense_layers=[128, 64]):
        
        self.train_dataset = tf.data.TFRecordDataset(TRAIN_SINGLE_PATH).map(lambda example_proto: _parse_image_function_2(example_proto, image_shape=IMAGE_SHAPE, label_shape=LABELS_SHAPE))
        self.val_dataset = tf.data.TFRecordDataset(TEST_SINGLE_PATH).map(lambda example_proto: _parse_image_function_2(example_proto, image_shape=IMAGE_SHAPE, label_shape=LABELS_SHAPE))
        
        self.model = mymodels.SingleModel(conv_layers=conv_layers, dense_layers=dense_layers)
        trainable_params = self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return trainable_params    
    
    def train(self, params=None):
        return super().train(SINGLE_CHECKPOINT_FOLDER, params)
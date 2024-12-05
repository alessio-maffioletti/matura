import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mymodels
import tensorflow as tf
from tensorflow.keras.backend import clear_session
import time

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
clear_session()


def _parse_image_function(example_proto, image_shape=[128,128,128,1], label_shape=[128,1]):
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

    # Ensure that the image tensor has the correct shape
    image.set_shape(image_shape)  # Set the known shape for the image tensor

    # Ensure that the label tensor has the correct shape
    label.set_shape(label_shape)

    return image, label

def _parse_image_function_2(example_proto, 
                          image_shape=[128, 128, 128, 1], 
                          coords_shape=[128, 2], 
                          label_shape=[128, 10]):
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

    # Set the correct shapes for the tensors
    image.set_shape(image_shape)    # Set the known shape for the image tensor
    coords.set_shape(coords_shape)  # Set the known shape for the coords tensor
    label.set_shape(label_shape)    # Set the known shape for the label tensor

    return (image, coords), label


class better_models:
    def __init__(self, main_folder='../', dataset_folder='dataset_tfrecord_small', checkpoints_folder=None, logs_folder=None):
        self.main_folder = main_folder   #main use
        #self.main_folder = './'     #for debugging
        self.dataset_folder = self.main_folder + dataset_folder + '/'
        if checkpoints_folder:
            self.checkpoints_folder = self.main_folder + checkpoints_folder + '/'
        else:
            self.checkpoints_folder = None
        if logs_folder:
            self.logs_folder = self.main_folder + logs_folder + '/'
        else:
            self.logs_folder = None

    def initialise_data_and_model(self, train_dataset_name, val_dataset_name, image_shape, label_shape, model_type, conv_layers=[32,64], dense_layers=[128,64], input_shape=(128, 128, 1), output_shape=10, activation='softmax'):
        clear_session()
        #'coords.tfrecord'
        #'coords_test.tfrecord'
        self.train_dataset = tf.data.TFRecordDataset(self.dataset_folder + train_dataset_name).map(lambda example_proto: _parse_image_function(example_proto, image_shape=image_shape, label_shape=label_shape))
        self.val_dataset = tf.data.TFRecordDataset(self.dataset_folder + val_dataset_name).map(lambda example_proto: _parse_image_function(example_proto, image_shape=image_shape, label_shape=label_shape))

        for a in self.train_dataset:
            image_batch, label_batch = a
            print(f"Image batch shape: {image_batch.shape}")  # Image batch
            print(f"Label batch shape: {label_batch.shape}")  # Label batch

    def train(self, params=None):
        start_time = time.time()
        
        self.run, reached_target = self.model.train(self.train_dataset, self.val_dataset, params, self.logs_folder, self.checkpoints_folder)
        
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
        print("The history has the following data: ", history_model.keys())

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
    def __init__(self, main_folder='../', dataset_folder='dataset_tfrecord_small', checkpoints_folder='checkpoints_sect1', logs_folder='logs'):
        super().__init__(main_folder, dataset_folder, checkpoints_folder, logs_folder)
    def initialise_data_and_model(self, train_dataset_name='coords.tfrecord', 
                                  val_dataset_name='coords_test.tfrecord', 
                                  model_type='regression', 
                                  image_shape=[128, 128, 128, 1], 
                                  label_shape=[128, 2], 
                                  conv_layers=[32, 64], 
                                  dense_layers=[128, 64], 
                                  input_shape=(128, 128, 1), 
                                  output_shape=2, 
                                  activation='linear'):
        
        super().initialise_data_and_model(train_dataset_name=train_dataset_name, val_dataset_name=val_dataset_name, image_shape=image_shape, label_shape=label_shape, model_type=model_type, conv_layers=conv_layers, dense_layers=dense_layers, input_shape=input_shape, output_shape=output_shape, activation=activation)
    
        self.model = mymodels.RegressionModel(conv_layers=conv_layers, dense_layers=dense_layers, input_shape=input_shape, output_shape=output_shape, activation=activation)
        trainable_params = self.model.compile()
        return trainable_params
        
    
    def _plot_random(self, image, predicted, actual):
        plt.imshow(image, cmap='gray')
        plt.scatter(predicted[0], predicted[1], color='red', label='Predicted')
        plt.scatter(actual[0], actual[1], color='blue', label='Actual')
        plt.legend()
        plt.title(f"Predicted: {predicted.round(1)} | Actual: {actual}")
        plt.show()
    
class section2(better_models):
    def __init__(self, main_folder='../', dataset_folder='dataset_tfrecord_small', checkpoints_folder='checkpoints_sect2', logs_folder='logs'):
        super().__init__(main_folder, dataset_folder, checkpoints_folder, logs_folder)
    def initialise_data_and_model(self, train_dataset_name='train_dataset_cropped.tfrecord', 
                                  val_dataset_name='test_dataset_cropped.tfrecord', 
                                  model_type='classification', 
                                  image_shape=[128,42,42,1], 
                                  label_shape=[128,10], 
                                  conv_layers=[32, 64], 
                                  dense_layers=[128, 64], 
                                  input_shape=(42, 42, 1), 
                                  output_shape=10, 
                                  activation='softmax'):
        
        super().initialise_data_and_model(train_dataset_name=train_dataset_name, val_dataset_name=val_dataset_name, image_shape=image_shape, label_shape=label_shape, model_type=model_type, conv_layers=conv_layers, dense_layers=dense_layers, input_shape=input_shape, output_shape=output_shape, activation=activation)
    
        self.model = mymodels.ClassificationModel(conv_layers=conv_layers, dense_layers=dense_layers, input_shape=input_shape, output_shape=output_shape, activation=activation)
        trainable_params = self.model.compile()

        return trainable_params
        
    
class single_model(better_models):
    def __init__(self, main_folder='../', dataset_folder='dataset_tfrecord_small', checkpoints_folder='checkpoints_single', logs_folder='logs'):
        super().__init__(main_folder, dataset_folder, checkpoints_folder, logs_folder)
    def initialise_data_and_model(self, train_dataset_name='train.tfrecord', 
                                  val_dataset_name='test.tfrecord', 
                                  model_type='single', 
                                  image_shape=[128, 128, 128, 1], 
                                  label_shape=[128, 10], 
                                  conv_layers=[32, 64], 
                                  dense_layers=[128, 64], 
                                  input_shape=(128, 128, 1), 
                                  output_shape=10, 
                                  activation='softmax'):
        
        self.train_dataset = tf.data.TFRecordDataset(self.dataset_folder + train_dataset_name).map(lambda example_proto: _parse_image_function_2(example_proto, image_shape=image_shape, label_shape=label_shape))
        self.val_dataset = tf.data.TFRecordDataset(self.dataset_folder + val_dataset_name).map(lambda example_proto: _parse_image_function_2(example_proto, image_shape=image_shape, label_shape=label_shape))
        self.model = mymodels.SingleModel()
        trainable_params = self.model.compile()
        return trainable_params    
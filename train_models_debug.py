import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mymodels
import tensorflow as tf
from tensorflow.keras.backend import clear_session

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

class models:
    def __init__(self):
        self.main_folder = '../'   #main use
        #self.main_folder = './'     #for debugging
        self.dataset_folder = self.main_folder + 'dataset2/'
        self.checkpoints_folder = self.main_folder + 'checkpoints_sect2/'
        self.logs_folder = self.main_folder + 'logs/'

    def initialise_data_and_model(self):
        clear_session()

        self.X = np.load(self.dataset_folder + '/X_cropped.npy')
        self.y = np.load(self.dataset_folder + '/y_train.npy')

        self.X_test = np.load(self.dataset_folder + '/X_cropped.npy')
        self.y_test = np.load(self.dataset_folder + '/y_train.npy')

        self.train_dataset = create_tf_data(self.X, self.y)
        self.val_dataset = create_tf_data(self.X_test, self.y_test)

        self.model = mymodels.sect2()
        self.model.compile()

    def train(self, params=None):
        if params == None:
            params = {'epochs': 30, 
                    'batch_size': 1024, 
                    'tensorboard': True, 
                    'cp_callback': True}
        self.run = self.model.train(self.train_dataset, self.val_dataset, params, self.logs_folder, self.checkpoints_folder)

    def plot(self):

        history_model = self.run.history
        print("The history has the following data: ", history_model.keys())

        fig, axs = plt.subplots(1, 2, figsize=(20, 5))

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
        
    def eval_random(self):
        #random_sample = np.random.randint(0, self.X_test.shape[0])
        for inp, lab in self.train_dataset.take(1):

            random_sample = np.random.randint(0, inp.shape[0])

            image = inp[random_sample][0]
            actual = lab[random_sample][0]
            prediction = self.model.predict(inp[random_sample])

            plt.imshow(image, cmap='gray')
            plt.title(f"Predicted: {prediction}")
            plt.show()

class sect2(models):
    def __init__(self):
        super().__init__()
        self.dataset_folder = self.main_folder + 'dataset2_small/'
        self.checkpoints_folder = self.main_folder + 'checkpoints_sect2/'

    def initialise_data_and_model(self):
        clear_session()

        self.train_dataset = tf.data.TFRecordDataset(self.dataset_folder + 'train_dataset_cropped.tfrecord').map(lambda example_proto: _parse_image_function(example_proto, image_shape=[128,42,42,1],label_shape=[128, 10]))
        self.val_dataset = tf.data.TFRecordDataset(self.dataset_folder + 'test_dataset_cropped.tfrecord').map(lambda example_proto: _parse_image_function(example_proto, image_shape=[128,42,42,1], label_shape=[128, 10]))
    

        self.model = mymodels.sect2()
        self.model.compile()



class sect1(models):
    def __init__(self):
        super().__init__()
        self.dataset_folder = self.main_folder + 'dataset_tfrecord_small/'
        self.checkpoints_folder = self.main_folder + 'checkpoints_sect1/'

    def initialise_data_and_model(self):
        clear_session()

        self.train_dataset = tf.data.TFRecordDataset(self.dataset_folder + 'coords.tfrecord').map(lambda example_proto: _parse_image_function(example_proto, label_shape=[128, 2]))
        self.val_dataset = tf.data.TFRecordDataset(self.dataset_folder + 'coords_test.tfrecord').map(lambda example_proto: _parse_image_function(example_proto, label_shape=[128, 2]))


        self.model = mymodels.sect1()
        self.model.compile()

class single(models):
    def __init__(self):
        super().__init__()
        self.dataset_folder = self.main_folder + 'dataset_tfrecord_small/'
        self.checkpoints_folder = self.main_folder + 'checkpoints_single/'

    def initialise_data_and_model(self):
        clear_session()

        self.train_dataset = tf.data.TFRecordDataset(self.dataset_folder + 'train.tfrecord').map(lambda example_proto: _parse_image_function(example_proto, label_shape=[128, 10]))
        self.val_dataset = tf.data.TFRecordDataset(self.dataset_folder + 'test.tfrecord').map(lambda example_proto: _parse_image_function(example_proto,label_shape=[128, 10]))


        # Initialize the model
        self.model = mymodels.single()
        self.model.compile()
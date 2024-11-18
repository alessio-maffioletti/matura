import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mymodels
import tensorflow as tf
from tensorflow.keras.backend import clear_session

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
clear_session()



def parse_tfrecord(example):
    """Parse a single example from the TFRecord file."""
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),  # The image as a byte string
        'label': tf.io.FixedLenFeature([10], tf.int64)  # The label (one-hot encoded)
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example['image'], channels=1)  # Decode the image
    label = example['label']
    image = tf.cast(image, tf.float32) / 255.0  # Normalize image
    return image, label
def load_tfrecord(filename, batch_size=32):
    """Load data from a TFRecord file and return a batched dataset."""
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_tfrecord)  # Parse the data
    dataset = dataset.batch(batch_size)  # Batch the data
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Improve performance by prefetching
    return dataset



def create_tf_data(X, y, batch_size=512, shuffle=False):
        """
        Converts NumPy arrays to a tf.data.Dataset.
        
        Parameters:
            X (np.array): Input features.
            y (np.array): Labels.
            batch_size (int): Batch size for training.
            shuffle (bool): Whether to shuffle the data.
            
        Returns:
            tf.data.Dataset: Batched and preprocessed dataset.
        """
        # Create a tf.data.Dataset from numpy arrays
        dataset = tf.data.Dataset.from_tensor_slices((X, y)).prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        #dataset = dataset.shuffle(buffer_size=10000).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
        #dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

class models:
    def __init__(self):
        self.main_folder = '../'
        #self.main_folder = './'
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
        random_sample = np.random.randint(0, self.X_test.shape[0])

        image = self.X_test[random_sample]
        actual = self.y_test[random_sample]
        prediction = self.model.predict(self.X_test[random_sample].reshape(1, 42, 42, 1))

        plt.imshow(image, cmap='gray')
        plt.title(f"Predicted: {prediction.argmax()}")
        plt.show()

class sect2(models):
    def __init__(self):
        super().__init__()

class sect1(models):
    def __init__(self):
        super().__init__()
        self.dataset_folder = self.main_folder + 'dataset_tfrecord_small/'
        self.checkpoints_folder = self.main_folder + 'checkpoints/'

    def initialise_data_and_model(self):
        clear_session()

        self.X = np.load(self.dataset_folder + '/X_train_canvas.npy')
        self.y = np.load(self.dataset_folder + '/coords.npy')

        self.X_test = np.load(self.dataset_folder + '/X_test_canvas.npy')
        self.y_test = np.load(self.dataset_folder + '/coords_test.npy')

        self.train_dataset = create_tf_data(self.X, self.y)
        self.val_dataset = create_tf_data(self.X_test, self.y_test)

        self.model = mymodels.sect1()
        self.model.compile()

class single(models):
    def __init__(self):
        super().__init__()
        self.dataset_folder = self.main_folder + 'dataset_tfrecord/'
        self.checkpoints_folder = self.main_folder + 'checkpoints/'

    def initialise_data_and_model(self):
        """Initialize the data and model."""
        self.train_dataset = load_tfrecord(self.dataset_folder + '/train_images_labels.tfrecord', batch_size=32)
        self.val_dataset = load_tfrecord(self.dataset_folder + '/test_images_labels.tfrecord', batch_size=32)

        # Initialize the model
        self.model = mymodels.single()
        self.model.compile()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mymodels
import tensorflow as tf
from tensorflow.keras.backend import clear_session

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
clear_session()


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

        # Plotting the training and validation accuracy during the training
        sns.lineplot(
            x=self.run.epoch, y=history_model[list(history_model)[0]], color="blue", label="Training set"
        )
        sns.lineplot(
            x=self.run.epoch,
            y=history_model[list(history_model)[2]],
            color="red",
            label="Valdation set",
        )
        plt.xlabel("epochs")
        plt.ylabel(list(history_model)[0])
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
        self.dataset_folder = self.main_folder + 'dataset/'
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
        self.dataset_folder = self.main_folder + 'dataset/'
        self.checkpoints_folder = self.main_folder + 'checkpoints/'

    def initialise_data_and_model(self, data=None):
        if data == None:
            self.X = np.load(self.dataset_folder + '/X_train_canvas.npy')
            self.y = np.load(self.dataset_folder + '/y_train.npy')

            self.X_test = np.load(self.dataset_folder + '/X_test_canvas.npy')
            self.y_test = np.load(self.dataset_folder + '/y_test.npy')
        else:
            self.X = data[0]
            self.y = data[1]

            self.X_test = data[2]
            self.y_test = data[3]

        self.train_dataset = create_tf_data(self.X[:1000], self.y[:1000])
        self.val_dataset = create_tf_data(self.X_test[:1000], self.y_test[:1000])

        self.model = mymodels.single()
        self.model.compile()
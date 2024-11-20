import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mymodels
import tensorflow as tf
from tensorflow.keras.backend import clear_session

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
clear_session()


def get_tensor(dataset):
    tensor= []
    for batch in dataset.take(-1):
        batch = np.array(batch)
        tensor.append(batch)

    tensor = np.array(tensor)
    return tensor

def load_tfrecord(filename, dataset_type=tf.int32):
    parse_tensor = lambda x: tf.io.parse_tensor(x, dataset_type)
    return tf.data.TFRecordDataset(filename).map(parse_tensor)
    
def make_tf_dataset(X, y):
    dataset = tf.data.Dataset.zip((X, y))
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

class models:
    def __init__(self):
        #self.main_folder = '../'   #main use
        self.main_folder = './'     #for debugging
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
        self.dataset_folder = self.main_folder + 'dataset2/'
        self.checkpoints_folder = self.main_folder + 'checkpoints_sect2/'

    def initialise_data_and_model(self):
        clear_session()

        self.X = load_tfrecord(self.dataset_folder + 'train_image_cropped.tfrecord', dataset_type=tf.double)
        self.y = load_tfrecord(self.dataset_folder + 'train_label.tfrecord', dataset_type=tf.float32)

        self.X_test = load_tfrecord(self.dataset_folder + 'test_image_cropped.tfrecord', dataset_type=tf.double)
        self.y_test = load_tfrecord(self.dataset_folder + 'test_label.tfrecord', dataset_type=tf.float32)

        self.train_dataset = make_tf_dataset(self.X, self.y)
        self.val_dataset = make_tf_dataset(self.X_test, self.y_test)


        self.model = mymodels.sect2()
        self.model.compile()

    def train_debug(self):
        # Create dummy input data with shape (batch_size, height, width, channels)
        input_data = tf.random.normal((128, 42, 42, 1))

        test_model = mymodels.sect2().model
        
        # Get the model output
        x = test_model(input_data)
        
        # Debug: Check the shape of the output tensor
        tf.debugging.assert_shapes([(x, (128, 42, 42, 1))])  # Assert that output has the expected shape
        
        # Optionally print the shape if it's needed for debugging
        print("Output shape:", x.shape)
        
        # Perform a simple forward pass and check for errors
        return x
    
    def train_debug2(self):
        # Example data

        x = get_tensor(self.X)
        y = get_tensor(self.y)

        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        # Train the model
        model = mymodels.sect2()
        model.compile()  # Compile the model first
        model.model.fit(dataset, epochs=10)



class sect1(models):
    def __init__(self):
        super().__init__()
        self.dataset_folder = self.main_folder + 'dataset_tfrecord/'
        self.checkpoints_folder = self.main_folder + 'checkpoints_sect1/'

    def initialise_data_and_model(self):
        clear_session()

        X = load_tfrecord(self.dataset_folder + 'train_image.tfrecord', dataset_type=tf.double)
        y = load_tfrecord(self.dataset_folder + 'train_coords.tfrecord', dataset_type=tf.int32)

        X_test = load_tfrecord(self.dataset_folder + 'test_image.tfrecord', dataset_type=tf.double)
        y_test = load_tfrecord(self.dataset_folder + 'test_coords.tfrecord', dataset_type=tf.int32)

        self.train_dataset = make_tf_dataset(X, y)
        self.val_dataset = make_tf_dataset(X_test, y_test)


        self.model = mymodels.sect1()
        self.model.compile()

class single(models):
    def __init__(self):
        super().__init__()
        self.dataset_folder = self.main_folder + 'dataset_tfrecord/'
        self.checkpoints_folder = self.main_folder + 'checkpoints_single/'

    def initialise_data_and_model(self):
        clear_session()

        X = load_tfrecord(self.dataset_folder + 'train_image.tfrecord', dataset_type=tf.double)
        y = load_tfrecord(self.dataset_folder + 'train_label.tfrecord', dataset_type=tf.int64)

        X_test = load_tfrecord(self.dataset_folder + 'test_image.tfrecord', dataset_type=tf.double)
        y_test = load_tfrecord(self.dataset_folder + 'test_label.tfrecord', dataset_type=tf.int64)

        self.train_dataset = make_tf_dataset(X, y)
        self.val_dataset = make_tf_dataset(X_test, y_test)

        # Initialize the model
        self.model = mymodels.single()
        self.model.compile()

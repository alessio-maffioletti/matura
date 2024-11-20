import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

#tf.config.set_visible_devices([], 'GPU')




# Loading the train and test data from the MNIST dataset

(X_train, y_train), (X_test, y_test) = mnist.load_data()

def place_image_on_canvas(image, width=1200, height=1200):
    """
    Place a single MNIST image at a random location on a blank canvas of given dimensions.
    """
    # Create a blank canvas
    canvas = np.zeros((height, width))

    # Image dimensions
    img_height, img_width = image.shape

    # Ensure the image fits: choose random coordinates for the top-left corner
    max_x, max_y = width - img_width, height - img_height
    x, y = np.random.randint(0, max_x), np.random.randint(0, max_y)

    # Place the image on the canvas
    canvas[y:y+img_height, x:x+img_width] = image

    return canvas, (x, y)

def show_random_images_on_canvases(n_images):
    """
    Show n_images of MNIST placed randomly on individual 1200x1200 canvases.
    """
    for _ in range(n_images):
        # Randomly choose an image
        idx = np.random.randint(0, X_train.shape[0])
        image = X_train[idx]

        # Create a canvas with the image placed randomly
        canvas = place_image_on_canvas(image, width=128, height=128)

        # Display the canvas
        plt.figure(figsize=(5, 5))
        plt.imshow(canvas, cmap='gray')
        plt.axis('off')
        plt.show()


def create_dataset(list_img, list_labels, n_images=100):
    """
    Create a dataset of n_images of MNIST placed randomly on individual 1200x1200 canvases.
    """
    X = []
    #y = []
    coords_x = []
    coords_y = []
    labels = []
    for i in range(n_images):
        image = list_img[i]
        # Create a canvas with the image placed randomly
        canvas, (x, y) = place_image_on_canvas(image, width=128, height=128)
        canvas = canvas.reshape(128, 128, 1)
        #flat_canvas = canvas_norm.flatten()
        # Add the canvas to the dataset
        X.append(canvas)
        labels.append(list_labels[i])
        coords_x.append(x)
        coords_y.append(y)
    
    X = np.array(X)
    coords = np.array([coords_x, coords_y]).T
    #X = X.reshape(X.shape[0], 128, 128, 1)

    return X, coords, labels


def write(X, filename, batch_size=69):
    serialized = tf.io.serialize_tensor(X)
    record_file = filename

    with tf.io.TFRecordWriter(record_file) as writer:
        writer.write(serialized.numpy())

def write_in_batches(data, filename, batch_size=100):
    #print(data.shape)
    record_file = filename
    with tf.io.TFRecordWriter(record_file) as writer:
        # Iterate through the dataset in batches
        for i in range(0, len(data), batch_size):
            # Check if this batch is a full batch
            if i + batch_size <= len(data):
                batch = data[i:i + batch_size]
                serialized = tf.io.serialize_tensor(batch)
                writer.write(serialized.numpy())

class notebook:
    def get_data(self):
        self.X_train_canvas, self.coords, self.y_train = create_dataset(X_train, y_train, n_images=X_train.shape[0])
        self.X_test_canvas, self.coords_test, self.y_test = create_dataset(X_test, y_test, n_images=X_test.shape[0])

        #scaler = MinMaxScaler()

        #self.coords = scaler.fit_transform(coords)
        #self.coords_test = scaler.fit_transform(coords_test)

        self.y_train_onehot = utils.to_categorical(y_train, num_classes=10).astype(np.int64)
        self.y_test_onehot = utils.to_categorical(y_test, num_classes=10).astype(np.int64)

        #print(self.X_train_canvas.shape, self.coords.shape, self.y_train_onehot.shape, self.X_test_canvas.shape, self.coords_test.shape, self.y_test_onehot.shape)

        return self.X_train_canvas, self.coords, self.y_train_onehot, self.X_test_canvas, self.coords_test, self.y_test_onehot
    
    def save_data(self):
        write_in_batches(self.X_train_canvas[:1000], 'train_image.tfrecord', batch_size=128)
        write_in_batches(self.coords[:1000], 'train_coords.tfrecord', batch_size=128)
        write_in_batches(self.y_train_onehot[:1000], 'train_label.tfrecord', batch_size=128)

        write_in_batches(self.X_test_canvas[:1000], 'test_image.tfrecord', batch_size=128)
        write_in_batches(self.coords_test[:1000], 'test_coords.tfrecord', batch_size=128)
        write_in_batches(self.y_test_onehot[:1000], 'test_label.tfrecord', batch_size=128)

if __name__ == '__main__':
    nt = notebook()
    X_train_canvas, coords, y_train_onehot, X_test_canvas, coords_test, y_test_onehot = nt.get_data()
    nt.save_data()
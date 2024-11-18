import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils



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

X_train_canvas, coords, y_train = create_dataset(X_train, y_train, n_images=X_train.shape[0]) #X_train.shape[0]
X_test_canvas, coords_test, y_test = create_dataset(X_test, y_test, n_images=X_test.shape[0]) #X_test.shape[0]

y_train_onehot = utils.to_categorical(y_train, num_classes=10).astype(np.int64)
y_test_onehot = utils.to_categorical(y_test, num_classes=10).astype(np.int64)

import tensorflow as tf

def serialize_example_image_label(image, label):
    """Convert an image and its label to a tf.train.Example."""
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(image).numpy()])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_example_coords_label(coords, label):
    """Convert coordinates and label to a tf.train.Example."""
    feature = {
        'coords_x': tf.train.Feature(int64_list=tf.train.Int64List(value=[coords[0]])),
        'coords_y': tf.train.Feature(int64_list=tf.train.Int64List(value=[coords[1]])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def save_to_tfrecord(X, y, filename, serialize_fn):
    """Save data to a TFRecord file using the provided serialize function."""
    with tf.io.TFRecordWriter(filename) as writer:
        for data, label in zip(X, y):
            tf_example = serialize_fn(data, label)
            writer.write(tf_example)


X_train_canvas, coords, y_train = create_dataset(X_train, y_train, n_images=X_train.shape[0])
X_test_canvas, coords_test, y_test = create_dataset(X_test, y_test, n_images=X_test.shape[0])

y_train_onehot = utils.to_categorical(y_train, num_classes=10).astype(np.int64)
y_test_onehot = utils.to_categorical(y_test, num_classes=10).astype(np.int64)

# Save training and testing data to separate TFRecord files
save_to_tfrecord(X_train_canvas, y_train_onehot[:1000], 'train_images_labels.tfrecord', serialize_example_image_label)
save_to_tfrecord(coords, y_train_onehot[:1000], 'train_coords_labels.tfrecord', serialize_example_coords_label)

save_to_tfrecord(X_test_canvas, y_test_onehot[:1000], 'test_images_labels.tfrecord', serialize_example_image_label)
save_to_tfrecord(coords_test, y_test_onehot[:1000], 'test_coords_labels.tfrecord', serialize_example_coords_label)
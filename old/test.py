import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def parse_tfrecord(example, dataset_type='image_coords'):
    """Parse a single example from the TFRecord file."""
    # Define feature description based on dataset type
    if dataset_type == 'image_label':
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),  # The image as a byte string
            'label': tf.io.FixedLenFeature([10], tf.int64)  # The label (one-hot encoded)
        }
    elif dataset_type == 'image_coords':
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),  # The image as a byte string
            'coords_x': tf.io.FixedLenFeature([1], tf.int64),  # x-coordinate
            'coords_y': tf.io.FixedLenFeature([1], tf.int64)   # y-coordinate
        }
    
    # Parse the example according to the feature description
    example = tf.io.parse_single_example(example, feature_description)
    
    # Decode the image
    image = tf.io.decode_jpeg(example['image'], channels=1)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize image
    
    if dataset_type == 'image_label':
        label = example['label']
        return image, label
    elif dataset_type == 'image_coords':
        coords = tf.stack([example['coords_x'], example['coords_y']], axis=-1)  # Stack coords into (x, y)
        return image, coords

def load_tfrecord(filename, batch_size=32, dataset_type='image_label'):
    """Load data from a TFRecord file and return a batched dataset."""
    # Load the TFRecord file
    dataset = tf.data.TFRecordDataset(filename)
    
    # Parse the data according to the dataset type
    dataset = dataset.map(lambda x: parse_tfrecord(x, dataset_type))
    
    # Batch the data
    dataset = dataset.batch(batch_size)
    
    # Prefetch the data for improved performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Set file paths (adjust as needed)
#main_folder = '../'   #main use
main_folder = './'     #for debugging
dataset_folder = main_folder + 'dataset_tfrecord_small/'
checkpoints_folder = main_folder + 'checkpoints_sect2/'
logs_folder = main_folder + 'logs/'

# Load train dataset
train_dataset = load_tfrecord(dataset_folder + '/train_image_coords.tfrecord', batch_size=128, dataset_type='image_coords')

# Fetch one batch from the dataset (you can change the batch size if needed)
for images, coords in train_dataset.take(1):  # Take the first batch
    image = images[0].numpy()  # Get the first image in the batch
    coord = coords[0].numpy()  # Get the first coordinate in the batch
    
    # Plot the image
    plt.imshow(image, cmap='gray')  # Squeeze removes the singleton dimension (channels)
    plt.title(f'Coordinates: x={coord[0][0]}, y={coord[0][1]}')  # Display coordinates
    plt.axis('off')  # Hide axes for better visual appeal
    plt.show()

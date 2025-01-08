import tensorflow as tf
import numpy as np

# Function to create a dummy dataset (replace with your actual dataset creation logic)
def create_dataset(batch_size=128):
    # Example: Generating random images (128x128x3) and random labels (one-hot)
    num_samples = 1000  # Example number of samples
    images = np.random.rand(num_samples, 128, 128, 3)  # Random images
    labels = np.random.randint(0, 10, size=(num_samples,))  # Random labels (0-9 classes)

    # Creating batches of data
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(batch_size)
    return dataset

# Write the dataset to a TFRecord file
def write_in_batches(dataset, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for images, labels in dataset:
            for image, label in zip(images.numpy(), labels.numpy()):
                # Create tf.train.Example for each image-label pair
                tf_example = example_test(image, label)
                writer.write(tf_example.SerializeToString())

# Function to serialize an image and label into tf.train.Example
def example_test(image, label):
    feature = {
        'image': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(tf.convert_to_tensor(image)).numpy()])
        ),
        'label': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[label])  # Label as an integer
        ),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto

# Main workflow
dataset_folder = './'
batch_size = 128
dataset = create_dataset(batch_size=batch_size)

# Save the dataset to a TFRecord file
write_in_batches(dataset, dataset_folder + 'train.tfrecord')
print("Dataset saved to 'train.tfrecord'")

del dataset  # Optional: Delete the dataset from memory

import tensorflow as tf

# Define the parse function to decode the image and label from the TFRecord file
def _parse_image_function(example_proto):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),  # Expect the image as a raw string (JPEG encoded)
        'label': tf.io.FixedLenFeature([1], tf.int64),  # Expect the label as an integer
    }
    
    # Parse the input tf.train.Example proto using the dictionary
    parsed_features = tf.io.parse_single_example(example_proto, image_feature_description)
    
    # Decode the image from JPEG
    image = tf.io.decode_jpeg(parsed_features['image'], channels=3)  # Decode image (3 channels for RGB)
    
    # Extract the label (integer)
    label = parsed_features['label'][0]
    
    return image, label

# Inspect the saved TFRecord
def inspect_tfrecord(filename):
    raw_dataset = tf.data.TFRecordDataset(filenames=[filename])
    for raw_record in raw_dataset.take(1):  # Take a single record for inspection
        print("Raw Record:", raw_record.numpy())
        parsed_features = _parse_image_function(raw_record)
        print("Parsed Features:", parsed_features)

# Loading and parsing the TFRecord file
def load_tfrecord(filename, batch_size=128):
    dataset = tf.data.TFRecordDataset(filenames=[filename])
    parsed_dataset = dataset.map(_parse_image_function)  # Parse the examples
    
    # Shuffle and batch the dataset
    parsed_dataset = parsed_dataset.shuffle(buffer_size=10000)
    parsed_dataset = parsed_dataset.batch(batch_size)
    
    return parsed_dataset

# Load the dataset
dataset = load_tfrecord(dataset_folder + 'train.tfrecord', batch_size=batch_size)

# Inspect the dataset
inspect_tfrecord(dataset_folder + 'train.tfrecord')

# Example of using the dataset for training
# Define a simple model for demonstration (replace with your actual model)
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),  # Input layer for images
        tf.keras.layers.Flatten(),  # Flatten the image into a vector
        tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer
        tf.keras.layers.Dense(10, activation='softmax')  # Output layer for 10 classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and compile the model
model = create_model()

# Train the model using the parsed dataset
model.fit(dataset, epochs=10)

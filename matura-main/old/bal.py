import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import mymodels

# Generate a random dataset with input shape (128, 128, 1) and 10 classes
def generate_random_data(batch_size=128, image_shape=(128, 128, 1), num_classes=10):
    # Generate random images
    images = np.random.rand(batch_size, *image_shape).astype(np.int64)
    
    # Generate random labels in one-hot encoding format
    labels = np.random.randint(0, num_classes, batch_size)
    labels = tf.keras.utils.to_categorical(labels, num_classes)
    
    return images, labels

# Convert to tf.data.Dataset for efficient batching
def create_dataset_old(batch_size=128, image_shape=(128, 128, 1), num_classes=10):
    def generator():
        for _ in range(7):
            images, labels = generate_random_data(batch_size, image_shape, num_classes)
            yield images, labels
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, *image_shape), dtype=tf.int64),
            tf.TensorSpec(shape=(batch_size, num_classes), dtype=tf.int64)
        )
    )
    dataset = dataset.shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def create_dataset(batch_size=128):
    # Example: Generating random images (128x128x3) and random labels (one-hot)
    num_samples = 896  # Example number of samples
    images = np.random.rand(num_samples, 128, 128, 1)  # Random images
    labels = np.random.randint(0, 10, size=(num_samples,))  # Random labels (0-9 classes)

    # Creating batches of data
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(batch_size)
    return dataset


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  #if isinstance(value, type(tf.constant(0))):
   # value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    value = value.flatten()  # Flatten the entire batch into a 1D list
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))  # Convert to list

def _float_feature(value):
    value = value.flatten()  # Flatten the entire batch into a 1D list
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))




def example_test(image, label):
    feature = {
        'image': _bytes_feature(image),
        'label': _bytes_feature(label)
    }
    #print(f'feature: {feature["label"]}')
    return tf.train.Example(features=tf.train.Features(feature=feature))

# Write dataset to TFRecord
def write_in_batches(data, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for images, labels in data:
            # Serialize images and labels as tensors
            image_tesnor = tf.convert_to_tensor(images)
            print(image_tesnor.shape)

            serialized_image = tf.io.serialize_tensor(image_tesnor).numpy()
            #labels_unonehot = np.argmax(labels, axis=-1)  # Convert one-hot labels to class indices
            label_tensor = tf.reshape(tf.convert_to_tensor(labels), [-1])
            reshaped_tensor = tf.expand_dims(label_tensor, axis=-1)  # Adds a new dimension at the last axis
            print(reshaped_tensor.shape)

            serialized_label = tf.io.serialize_tensor(reshaped_tensor).numpy()

            
            # Create tf.train.Example and write it to the file
            tf_example = example_test(serialized_image, serialized_label)
            writer.write(tf_example.SerializeToString())

def _parse_image_function(example_proto):
    # Define the features to be extracted (serialized image and label)
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),  # Expecting the image as a serialized tensor (string)
        'label': tf.io.FixedLenFeature([], tf.string),  # Expecting the label as a serialized tensor (string)
    }

    # Parse the input tf.train.Example proto using the dictionary
    parsed_features = tf.io.parse_single_example(example_proto, image_feature_description)
    
    # Deserialize the image and label tensors
    image = tf.io.parse_tensor(parsed_features['image'], out_type=tf.double)  # Deserialize image tensor
    label = tf.io.parse_tensor(parsed_features['label'], out_type=tf.int32)  # Deserialize label tensor

    # Ensure that the image tensor has the correct shape
    image.set_shape([128, 128, 128, 1])  # Set the known shape for the image tensor

    # Ensure that the label tensor has the correct shape
    label.set_shape([128,1])

    return image, label






dataset_folder = './dataset/'

# Create a dataset
batch_size = 128
dataset = create_dataset(batch_size=batch_size)

# Create and compile the model
model = mymodels.debug()
model.compile()


# Train the model using the parsed dataset
model.model.fit(dataset, epochs=10)

print("sd")
write_in_batches(dataset, 'train.tfrecord')
print("ha")
del dataset

dataset = tf.data.TFRecordDataset('train.tfrecord')
parsed_dataset = dataset.map(_parse_image_function)

for data in parsed_dataset:
  a,b = data
  print(a.shape, b.shape)
  print(b)

model = mymodels.debug()
model.compile()
model.model.fit(parsed_dataset, epochs=10)
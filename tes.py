import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import mymodels

# Generate a random dataset with input shape (128, 128, 1) and 10 classes
def generate_random_data(batch_size=128, image_shape=(128, 128, 1), num_classes=10):
    # Generate random images
    images = np.random.rand(batch_size, *image_shape).astype(np.float32)
    
    # Generate random labels in one-hot encoding format
    labels = np.random.randint(0, num_classes, batch_size)
    labels = tf.keras.utils.to_categorical(labels, num_classes)
    
    return images, labels

# Convert to tf.data.Dataset for efficient batching
def create_dataset(batch_size=128, image_shape=(128, 128, 1), num_classes=10):
    dataset = tf.data.Dataset.from_generator(
        lambda: (generate_random_data(batch_size, image_shape, num_classes)),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, *image_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, num_classes), dtype=tf.float32)
        )
    )
    return dataset

# Initialize the model
model = mymodels.single()

# Compile the model
model.compile()

# Create a dataset
batch_size = 128
dataset = create_dataset(batch_size=batch_size)

# Train the model using the randomly generated dataset
model.model.fit(dataset, epochs=5, steps_per_epoch=10)
import tensorflow as tf
import time
from tensorflow.keras import mixed_precision

# mixed_precision.set_global_policy('mixed_float16')

# Define a helper function to measure training time
def train_model(device_name, epochs=5):
    """
    Train a simple neural network on a specific device (CPU or GPU)
    and measure the training time.

    Parameters:
        device_name (str): The device to use for training, e.g., '/CPU:0' or '/GPU:0'
        epochs (int): Number of training epochs

    Returns:
        dict: Training metrics including time, loss, and accuracy
    """
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the data

    # Convert to tf.data.Dataset for optimization
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Shuffle, batch, and prefetch data
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().shuffle(buffer_size=10000).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Define a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model on the specified device
    with tf.device(device_name):
        start_time = time.time()
        history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=1)
        end_time = time.time()

    # Calculate total training time
    training_time = end_time - start_time

    # Get the final loss and accuracy
    final_loss, final_accuracy = model.evaluate(test_dataset, verbose=1)

    return {
        "device": device_name,
        "training_time": training_time,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy
    }

# Main function to compare CPU and GPU performance
def compare_devices():
    print("Comparing CPU and GPU performance...")

    # Train on CPU
    cpu_metrics = train_model('/CPU:0', epochs=5)
    print(f"\nCPU Results:\n"
          f"Training Time: {cpu_metrics['training_time']:.2f} seconds\n"
          f"Final Loss: {cpu_metrics['final_loss']:.4f}\n"
          f"Final Accuracy: {cpu_metrics['final_accuracy']:.4f}")

    # Check if GPU is available
    if tf.config.list_physical_devices('GPU'):
        gpu_metrics = train_model('/GPU:0', epochs=5)
        print(f"\nGPU Results:\n"
              f"Training Time: {gpu_metrics['training_time']:.2f} seconds\n"
              f"Final Loss: {gpu_metrics['final_loss']:.4f}\n"
              f"Final Accuracy: {gpu_metrics['final_accuracy']:.4f}")
    else:
        print("\nNo GPU detected. Please ensure your TensorFlow installation is configured for GPU support.")

# Run the comparison
if __name__ == "__main__":
    compare_devices()

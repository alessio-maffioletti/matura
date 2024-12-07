import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import mymodels
from constants import *
import psutil

def limit_memory_windows(max_memory_gb):
    max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
    process = psutil.Process()
    if process.memory_info().rss > max_memory_bytes:
        raise MemoryError(f"Memory usage exceeds {max_memory_gb} GB")

# Example usage
limit_memory_windows(10)  # Raise an error if the process uses more than 2 GB of memory



class TFRecordHandler:
    def __init__(self, image_shape=(128, 128, 1), label_shape=(10,), coords_shape=None):
        self.image_shape = image_shape
        self.label_shape = label_shape
        self.coords_shape = coords_shape

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string/byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def serialize_example(self, image, label, coords=None):
        """Serialize a single example into a tf.train.Example."""
        feature = {
            'image': self._bytes_feature(image),
            'label': self._bytes_feature(label)
        }
        if coords is not None:
            feature['coords'] = self._bytes_feature(coords)

        return tf.train.Example(features=tf.train.Features(feature=feature))
    

    def write_tfrecord(self, data, filename, dual_input=False):
        """Write data to a TFRecord file."""
        with tf.io.TFRecordWriter(filename) as writer:
            for batch in data:
                if dual_input:
                    (images, coords), labels = batch
                    serialized_coords = tf.io.serialize_tensor(tf.convert_to_tensor(coords)).numpy()
                else:
                    images, labels = batch
                    serialized_coords = None

                serialized_image = tf.io.serialize_tensor(tf.convert_to_tensor(images)).numpy()
                serialized_label = tf.io.serialize_tensor(tf.convert_to_tensor(labels)).numpy()

                tf_example = self.serialize_example(serialized_image, serialized_label, serialized_coords)
                writer.write(tf_example.SerializeToString())

    def data_generator(self, images, labels, coords=None):
        """Generator for batching data."""
        for i in range(0, len(images), BATCH_SIZE):
            if coords is not None:
                yield (images[i:i+BATCH_SIZE], coords[i:i+BATCH_SIZE]), labels[i:i+BATCH_SIZE]
            else:
                yield images[i:i+BATCH_SIZE], labels[i:i+BATCH_SIZE]

    def create_tf_dataset(self, images, labels, coords=None):
        """Create a TensorFlow dataset."""
        if coords is not None:
            dataset = tf.data.Dataset.from_generator(
                lambda: self.data_generator(images, labels, coords),
                output_signature=(
                    (
                        tf.TensorSpec(shape=(BATCH_SIZE, *self.image_shape), dtype=tf.float32),
                        tf.TensorSpec(shape=(BATCH_SIZE, *self.coords_shape), dtype=tf.float32)
                    ),
                    tf.TensorSpec(shape=(BATCH_SIZE, *self.label_shape), dtype=tf.int32)
                )
            )
        else:
            dataset = tf.data.Dataset.from_generator(
                lambda: self.data_generator(images, labels),
                output_signature=(
                    tf.TensorSpec(shape=(BATCH_SIZE, *self.image_shape), dtype=tf.float32),
                    tf.TensorSpec(shape=(BATCH_SIZE, *self.label_shape), dtype=tf.int32)
                )
            )
        return dataset
    

class Dataset1:
    def __init__(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        self.y_train = utils.to_categorical(self.y_train, 10)
        self.y_test = utils.to_categorical(self.y_test, 10)

    def place_image_on_canvas(self, image, width=120, height=120):
        canvas = np.zeros((height, width))

        img_height, img_width = image.shape

        max_x, max_y = width - img_width, height - img_height
        x, y = np.random.randint(0, max_x), np.random.randint(0, max_y)

        canvas[y:y+img_height, x:x+img_width] = image

        return canvas, (x, y)


    def create_dataset(self, list_img, list_labels, n_images=100):
        X = []
        coords_x = []
        coords_y = []
        labels = []
        for i in range(n_images):
            image = list_img[i]
            # Create a canvas with the image placed randomly
            canvas, (x, y) = self.place_image_on_canvas(image, width=128, height=128)
            canvas = canvas.reshape(128, 128, 1)
            #flat_canvas = canvas_norm.flatten()
            # Add the canvas to the dataset
            X.append(canvas)
            labels.append(list_labels[i])
            coords_x.append(x)
            coords_y.append(y)
        
        X = np.array(X)
        coords = np.array([coords_x, coords_y]).T

        return X, coords, labels
    
    def generate_dataset(self, train_size=1280, test_size=1280, whole_dataset=False):
        if whole_dataset:
            images, coords, labels = self.create_dataset(self.X_train, self.y_train, n_images=self.X_train.shape[0]) #59904
            images_test, coords_test, labels_test = self.create_dataset(self.X_test, self.y_test, n_images=self.X_test.shape[0]) #9984
        else:
            if train_size % BATCH_SIZE != 0:
                train_size -= train_size % BATCH_SIZE
            if test_size % BATCH_SIZE != 0:
                test_size -= test_size % BATCH_SIZE
        
            images, coords, labels = self.create_dataset(self.X_train, self.y_train, n_images=train_size) #59904
            images_test, coords_test, labels_test = self.create_dataset(self.X_test, self.y_test, n_images=test_size) #9984

        single_handler = TFRecordHandler(image_shape=INPUT_SHAPE, label_shape=COORDS_SHAPE)
        coords_dataset = single_handler.create_tf_dataset(images, coords)
        single_handler.write_tfrecord(coords_dataset, filename=TRAIN_COORDS_PATH, dual_input=False)
        del coords_dataset
        coords_test_dataset = single_handler.create_tf_dataset(images_test, coords_test)
        single_handler.write_tfrecord(coords_test_dataset, filename=TEST_COORDS_PATH, dual_input=False)
        del coords_test_dataset

        dual_handler = TFRecordHandler(image_shape=INPUT_SHAPE, label_shape=LABELS_SHAPE, coords_shape=COORDS_SHAPE)
        train_dataset = dual_handler.create_tf_dataset(images, labels, coords=coords)
        dual_handler.write_tfrecord(train_dataset, filename=TRAIN_SINGLE_PATH, dual_input=True)
        del train_dataset
        test_dataset = dual_handler.create_tf_dataset(images_test, labels_test, coords=coords_test)
        dual_handler.write_tfrecord(test_dataset, filename=TEST_SINGLE_PATH, dual_input=True)
        del test_dataset

class Dataset2:
    def __init__(self, weights, conv_layers, dense_layers):
        self.weights = weights
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers

    def _parse_image_function(self,
                                example_proto, 
                                image_shape=[128, 128, 1], 
                                coords_shape=[2], 
                                label_shape=[10]):
        # Define the features to be extracted (serialized tensors for image, coords, and label)
        image_feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),   # Serialized image tensor
            'coords': tf.io.FixedLenFeature([], tf.string),  # Serialized coords tensor
            'label': tf.io.FixedLenFeature([], tf.string),   # Serialized label tensor
        }

        # Parse the input `tf.train.Example` proto using the dictionary
        parsed_features = tf.io.parse_single_example(example_proto, image_feature_description)
        
        # Deserialize the tensors
        image = tf.io.parse_tensor(parsed_features['image'], out_type=tf.float32)  # Deserialize image tensor
        coords = tf.io.parse_tensor(parsed_features['coords'], out_type=tf.float32)  # Deserialize coords tensor
        label = tf.io.parse_tensor(parsed_features['label'], out_type=tf.int32)  # Deserialize label tensor

        image_shape = [BATCH_SIZE, *image_shape]
        coords_shape = [BATCH_SIZE, *coords_shape]
        label_shape = [BATCH_SIZE, *label_shape]

        # Set the correct shapes for the tensors
        image.set_shape(image_shape)    # Set the known shape for the image tensor
        coords.set_shape(coords_shape)  # Set the known shape for the coords tensor
        label.set_shape(label_shape)    # Set the known shape for the label tensor

        return (image, coords), label
    
    def read_tfrecord(self, filename):
        dataset = tf.data.TFRecordDataset(filename).map(lambda example_proto: self._parse_image_function(example_proto))
        return dataset
    
    def crop_images(self, X_train_canvas, batch, model):
        X_cropped = []
        crop_amount = 32
        base_grace = 5

        small_data = X_train_canvas #[:100]

        for i in range(small_data.shape[0]):
            grace = base_grace

            image = small_data[i]
            coords = model.predict(image.reshape(1,128,128,1))
            x = int(coords[0][0])
            y = int(coords[0][1])
            if x > 128-crop_amount-grace:
                x = 128-crop_amount-grace
            if x < grace:
                x = grace
            if y > 128-crop_amount-grace:
                y = 128-crop_amount-grace
            if y < grace:
                y = grace
            
            cropped_image = image[y-grace:y+crop_amount+grace, x-grace:x+crop_amount+grace]

            if cropped_image.shape != (42,42,1):
                print(f"Error: {cropped_image.shape}")
                print(x,y)
                plt.imshow(cropped_image)
                plt.show()
                plt.imshow(image)
                plt.show()
            X_cropped.append(cropped_image)
            print(f"\rNum: {i+1} / {small_data.shape[0]} Batch: {batch} ", end='')

        X_cropped = np.array(X_cropped)
        print(X_cropped.shape)
        return X_cropped
    
    def _bytes_feature(sefl, value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def example_test(self, image, label):
        feature = {
            'image': self._bytes_feature(image),
            'label': self._bytes_feature(label)
        }
        #print(f'feature: {feature["label"]}')
        return tf.train.Example(features=tf.train.Features(feature=feature))
    
    def write_tfrecord(self, data, model, filename):
        """Write data to a TFRecord file."""
        with tf.io.TFRecordWriter(filename) as writer:
            for n, batch in enumerate(data):
                (images, _), labels = batch
                images = self.crop_images(images.numpy(), n, model)
                serialized_image = tf.io.serialize_tensor(tf.convert_to_tensor(images)).numpy()
                serialized_label = tf.io.serialize_tensor(tf.convert_to_tensor(labels)).numpy()

                tf_example = self.example_test(serialized_image, serialized_label)
                writer.write(tf_example.SerializeToString())

    def generate_dataset(self):
        train_dataset = self.read_tfrecord(filename=TRAIN_SINGLE_PATH)
        val_dataset = self.read_tfrecord(filename=TEST_SINGLE_PATH)

        model = mymodels.RegressionModel(INPUT_SHAPE, COORDS_OUTPUT_SHAPE, REGRESSION_ACTIVATION, self.conv_layers, self.dense_layers)
        model.compile()
        model.load_weights(self.weights)

        self.write_tfrecord(train_dataset, model, filename=TRAIN_CROPPED_PATH)
        self.write_tfrecord(val_dataset, model, filename=TEST_CROPPED_PATH)



if __name__ == "__main__":
    #dataset1_generator = Dataset1()
    #dataset1_generator.generate_dataset()

    print("hehe")

    #dataset2_generator = Dataset2('./checkpoints_sect1/modelfinal08.weights.h5')
    #dataset2_generator.generate_dataset()


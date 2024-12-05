import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import mymodels


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

    def data_generator(self, images, labels, coords=None, batch_size=128):
        """Generator for batching data."""
        for i in range(0, len(images), batch_size):
            if coords is not None:
                yield (images[i:i+batch_size], coords[i:i+batch_size]), labels[i:i+batch_size]
            else:
                yield images[i:i+batch_size], labels[i:i+batch_size]

    def create_tf_dataset(self, images, labels, coords=None, batch_size=128):
        """Create a TensorFlow dataset."""
        if coords is not None:
            dataset = tf.data.Dataset.from_generator(
                lambda: self.data_generator(images, labels, coords, batch_size),
                output_signature=(
                    (
                        tf.TensorSpec(shape=(batch_size, *self.image_shape), dtype=tf.float32),
                        tf.TensorSpec(shape=(batch_size, *self.coords_shape), dtype=tf.float32)
                    ),
                    tf.TensorSpec(shape=(batch_size, *self.label_shape), dtype=tf.int32)
                )
            )
        else:
            dataset = tf.data.Dataset.from_generator(
                lambda: self.data_generator(images, labels, batch_size=batch_size),
                output_signature=(
                    tf.TensorSpec(shape=(batch_size, *self.image_shape), dtype=tf.float32),
                    tf.TensorSpec(shape=(batch_size, *self.label_shape), dtype=tf.int32)
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
    
    def generate_dataset(self,train_coords_path, test_coords_path, train_labels_path, test_labels_path,  n_images=1280, ):
        images, coords, labels = self.create_dataset(self.X_train, self.y_train, n_images=n_images) #59904
        images_test, coords_test, labels_test = self.create_dataset(self.X_test, self.y_test, n_images=n_images) #9984

        single_handler = TFRecordHandler(image_shape=(128, 128, 1), label_shape=(2,))
        coords_dataset = single_handler.create_tf_dataset(images, coords)
        coords_test_dataset = single_handler.create_tf_dataset(images_test, coords_test)
        single_handler.write_tfrecord(coords_dataset, train_coords_path, dual_input=False)
        single_handler.write_tfrecord(coords_test_dataset, test_coords_path, dual_input=False)

        dual_handler = TFRecordHandler(image_shape=(128, 128, 1), label_shape=(10,), coords_shape=(2,))
        train_dataset = dual_handler.create_tf_dataset(images, labels, coords=coords)
        test_dataset = dual_handler.create_tf_dataset(images_test, labels_test, coords=coords_test)
        dual_handler.write_tfrecord(train_dataset, train_labels_path, dual_input=True)
        dual_handler.write_tfrecord(test_dataset, test_labels_path, dual_input=True)

class Dataset2:
    def __init__(self, weights):
        self.weights = weights

    def _parse_image_function(self, example_proto, image_shape=[128, 128, 128, 1], coords_shape=[128, 2], label_shape=[128, 10]):
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

        # Set the correct shapes for the tensors
        image.set_shape(image_shape)    # Set the known shape for the image tensor
        coords.set_shape(coords_shape)  # Set the known shape for the coords tensor
        label.set_shape(label_shape)    # Set the known shape for the label tensor

        return (image, coords), label
    
    def read_tfrecord(self, filename):
        dataset = tf.data.TFRecordDataset(filename).map(self._parse_image_function)
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
    
    def write_tfrecord(self, data, filename, model):
        """Write data to a TFRecord file."""
        with tf.io.TFRecordWriter(filename) as writer:
            for n, batch in enumerate(data):
                (images, _), labels = batch
                images = self.crop_images(images.numpy(), n, model)
                serialized_image = tf.io.serialize_tensor(tf.convert_to_tensor(images)).numpy()
                serialized_label = tf.io.serialize_tensor(tf.convert_to_tensor(labels)).numpy()

                tf_example = self.example_test(serialized_image, serialized_label)
                writer.write(tf_example.SerializeToString())

    def generate_dataset(self, train_dataset_path, test_dataset_path, write_train_dataset_path, write_test_dataset_path):
        train_dataset = self.read_tfrecord(train_dataset_path)
        val_dataset = self.read_tfrecord(test_dataset_path)

        model = mymodels.RegressionModel()
        model.compile()
        model.load_weights(self.weights)

        self.write_tfrecord(train_dataset, write_train_dataset_path, model)
        self.write_tfrecord(val_dataset, write_test_dataset_path, model)



if __name__ == "__main__":
    #dataset1_generator = Dataset1()
    #dataset1_generator.generate_dataset()

    print("hehe")

    #dataset2_generator = Dataset2('./checkpoints_sect1/modelfinal08.weights.h5')
    #dataset2_generator.generate_dataset()


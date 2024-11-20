import matplotlib.pyplot as plt
import numpy as np
import mymodels
import tensorflow as tf

main_folder = './'
dataset_folder = main_folder + 'dataset_tfrecord_small/'
checkpoints_folder = main_folder + 'checkpoints_sect1/'


def write(X, filename, batch_size=3):
    serialized = tf.io.serialize_tensor(X)
    record_file = filename

    with tf.io.TFRecordWriter(record_file) as writer:
        writer.write(serialized.numpy())

def write_and_crop_in_batches(data, filename, batch_size=128):
    record_file = filename
    with tf.io.TFRecordWriter(record_file) as writer:
        for i, batch in enumerate(data.take(-1)):
            batch = np.array(batch)
            batch = crop_images(batch, i)
            if batch.shape != (batch_size, 42, 42, 1):
                print(f"Error: {batch.shape}")
            serialized = tf.io.serialize_tensor(batch)
            writer.write(serialized.numpy())

def load_tfrecord(filename, dataset_type=tf.int32):
    parse_tensor = lambda x: tf.io.parse_tensor(x, dataset_type)
    return tf.data.TFRecordDataset(filename).map(parse_tensor)

def get_tensor(x):
    whole_tensor = []
    for batch in x:
        np_batch = np.array(batch)
        whole_tensor.append(np_batch)
    whole_tensor = np.array(whole_tensor)
    return whole_tensor

def batched_crop(x):
    whole_tensor = []
    batch_i = 0
    for batch in x:
        np_batch = np.array(batch)
        whole_tensor.append(crop_images(np_batch, batch_i))
        batch_i += 1
    whole_tensor = np.array(whole_tensor)
    return whole_tensor
    
        
    
def make_tf_dataset(X, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def crop_images(X_train_canvas, batch):
    X_cropped = []
    crop_amount = 32
    base_grace = 5

    small_data = X_train_canvas #[:100]

    for i in range(small_data.shape[0]):
        grace = base_grace

        image = small_data[i]
        coords = model.predict(image.reshape(1, 128, 128, 1))
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
    return X_cropped


weights = checkpoints_folder + 'sect1_epoch_20.weights.h5'
model = mymodels.sect1()
model.compile()
model.load_weights(weights)

X_train = load_tfrecord(dataset_folder + 'train_image.tfrecord', dataset_type=tf.double)
#y = load_tfrecord(dataset_folder + 'train_label.tfrecord', dataset_type=tf.int64)

X_test = load_tfrecord(dataset_folder + 'test_image.tfrecord', dataset_type=tf.double)
#y_test = load_tfrecord(dataset_folder + 'test_label.tfrecord', dataset_type=tf.int64)

write_and_crop_in_batches(X_train, 'train_image_cropped.tfrecord', batch_size=128)
write_and_crop_in_batches(X_test, 'test_image_cropped.tfrecord', batch_size=128)

#y_train = get_tensor(y)
#y_test = get_tensor(y_test)



'''
X_cropped = batched_crop(X)
X_test_cropped = batched_crop(X_test)

#y_train = tf.stack(y_train)
#y_test = tf.stack(y_test)
#X_cropped = tf.stack(X_cropped)
#X_test_cropped = tf.stack(X_test_cropped)

print(X_cropped.shape, y_train.shape, X_test_cropped.shape, y_test.shape)



#X_cropped = crop_images(X_train_canvas)
#X_test_cropped = crop_images(X_test_canvas)

write(X_cropped, 'train_image_cropped.tfrecord', batch_size=128)
write(y_train, 'train_label.tfrecord', batch_size=128)

write(X_test_cropped, 'test_image_cropped.tfrecord', batch_size=128)
write(y_test, 'test_label.tfrecord', batch_size=128)
'''
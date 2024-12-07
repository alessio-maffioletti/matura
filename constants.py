main_folder = '../'
dataset1_folder = main_folder + 'dataset_tfrecord_small' + '/'
dataset2_folder = main_folder + 'dataset2_small' + '/'

TRAIN_SINGLE_PATH = dataset1_folder + 'train.tfrecord'
TEST_SINGLE_PATH = dataset1_folder + 'test.tfrecord'

TRAIN_COORDS_PATH = dataset1_folder + 'coords.tfrecord' 
TEST_COORDS_PATH = dataset1_folder + 'coords_test.tfrecord'

TRAIN_CROPPED_PATH = dataset2_folder + 'train_cropped.tfrecord',
TEST_CROPPED_PATH = dataset2_folder + 'test_cropped.tfrecord',

BATCH_SIZE = 32


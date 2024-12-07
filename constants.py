main_folder = '../'
dataset1_folder = main_folder + 'dataset_tfrecord_small' + '/'
dataset2_folder = main_folder + 'dataset2_small' + '/'

TRAIN_SINGLE_PATH = dataset1_folder + 'train.tfrecord'
TEST_SINGLE_PATH = dataset1_folder + 'test.tfrecord'

TRAIN_COORDS_PATH = dataset1_folder + 'coords.tfrecord' 
TEST_COORDS_PATH = dataset1_folder + 'coords_test.tfrecord'

TRAIN_CROPPED_PATH = dataset2_folder + 'train_cropped.tfrecord'
TEST_CROPPED_PATH = dataset2_folder + 'test_cropped.tfrecord'

SECT1_CHECKPOINT_FOLDER = main_folder + 'checkpoints_sect1' + '/'
SECT2_CHECKPOINT_FOLDER = main_folder + 'checkpoints_sect2' + '/'
SINGLE_CHECKPOINT_FOLDER = main_folder + 'checkpoints_single' + '/'

LOGS_FOLDER = main_folder + 'logs' + '/'

BATCH_SIZE = 32

IMAGE_SHAPE = [128,128,1]
CROPPED_IMAGE_SHAPE = [42,42,1]
COORDS_SHAPE = [2]
LABELS_SHAPE = [10]

INPUT_SHAPE = tuple(IMAGE_SHAPE)
CROPPED_INPUT_SHAPE = tuple(CROPPED_IMAGE_SHAPE)
COORDS_OUTPUT_SHAPE = int(COORDS_SHAPE[0])
LABELS_OUTPUT_SHAPE = int(LABELS_SHAPE[0])

CLASSIFICATION_ACTIVATION = 'softmax'
REGRESSION_ACTIVATION = 'linear'
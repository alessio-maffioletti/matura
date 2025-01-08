main_folder = '../'
dataset1_folder = main_folder + 'dataset_tfrecord' + '/'
dataset2_folder = main_folder + 'dataset2' + '/'

TRAIN_SINGLE_PATH = dataset1_folder + 'train.tfrecord'
TEST_SINGLE_PATH = dataset1_folder + 'test.tfrecord'

TRAIN_COORDS_PATH = dataset1_folder + 'coords.tfrecord' 
TEST_COORDS_PATH = dataset1_folder + 'coords_test.tfrecord'

TRAIN_CROPPED_PATH = dataset2_folder + 'train_cropped.tfrecord'
TEST_CROPPED_PATH = dataset2_folder + 'test_cropped.tfrecord'

SECT1_CHECKPOINT_FOLDER = main_folder + 'checkpoints_sect1' + '/'
SECT2_CHECKPOINT_FOLDER = main_folder + 'checkpoints_sect2' + '/'
SINGLE_CHECKPOINT_FOLDER = main_folder + 'checkpoints_single' + '/'

OPTIMIZER_FOLDER = main_folder + 'optimizers' + '/'

LOGS_FOLDER = main_folder + 'logs' + '/'

BATCH_SIZE = 64

MAX_TRAIN_TIME = 300

RANDOM_SEED = 42

IMAGE_SHAPE = [128,128,1]
CROPPED_IMAGE = {
    'image_shape': [34,34,1],
    'input_shape': (34,34,1)
}
COORDS_SHAPE = [2]
LABELS_SHAPE = [10]

INPUT_SHAPE = tuple(IMAGE_SHAPE)
#CROPPED_INPUT_SHAPE = tuple(CROPPED_IMAGE_SHAPE['input_shape'])
COORDS_OUTPUT_SHAPE = int(COORDS_SHAPE[0])
LABELS_OUTPUT_SHAPE = int(LABELS_SHAPE[0])

CLASSIFICATION_ACTIVATION = 'softmax'
REGRESSION_ACTIVATION = 'linear'

def change_cropped_image_shape(image_shape):
    CROPPED_IMAGE['image_shape'] = image_shape
    CROPPED_IMAGE['input_shape'] = tuple(image_shape)
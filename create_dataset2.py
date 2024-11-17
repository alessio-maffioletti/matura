import matplotlib.pyplot as plt
import numpy as np
import mymodels

main_folder = './matura-private-main/'
dataset_folder = main_folder + 'dataset/'
dataset2_folder = main_folder + 'dataset2/'
logs_folder = main_folder + 'logs/'
checkpoints_folder = main_folder + 'checkpoints/'

X_train_canvas = np.load(dataset_folder + 'X_train_canvas.npy')
X_test_canvas = np.load(dataset_folder + 'X_test_canvas.npy')
y_train = np.load(dataset_folder + 'y_train.npy')
y_test = np.load(dataset_folder + 'y_test.npy')


weights = checkpoints_folder + 'model_epoch_30.weights.h5'
model = mymodels.sect1()
model.compile()
model.load_weights(weights)


X_cropped = []
crop_amount = 32
base_grace = 5

DATA_SIZE = 1000

small_data = X_train_canvas[:DATA_SIZE]

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

    if cropped_image.shape != (42,42):
        print(f"Error: {cropped_image.shape}")
        print(x,y)
        plt.imshow(cropped_image)
        plt.show()
        plt.imshow(image)
        plt.show()
    X_cropped.append(cropped_image)
    print(f"\rNum: {i+1} / {small_data.shape[0]}", end='')

X_cropped = np.array(X_cropped)
X_cropped = X_cropped.reshape(X_cropped.shape[0], 42, 42)


np.save(dataset2_folder + 'X_cropped.npy', X_cropped)
np.save(dataset2_folder + 'y_train.npy', y_train[:DATA_SIZE])
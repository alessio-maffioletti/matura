import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils



# Loading the train and test data from the MNIST dataset

(X_train, y_train), (X_test, y_test) = mnist.load_data()

def place_image_on_canvas(image, width=1200, height=1200):
    """
    Place a single MNIST image at a random location on a blank canvas of given dimensions.
    """
    # Create a blank canvas
    canvas = np.zeros((height, width))

    # Image dimensions
    img_height, img_width = image.shape

    # Ensure the image fits: choose random coordinates for the top-left corner
    max_x, max_y = width - img_width, height - img_height
    x, y = np.random.randint(0, max_x), np.random.randint(0, max_y)

    # Place the image on the canvas
    canvas[y:y+img_height, x:x+img_width] = image

    return canvas, (x, y)

def show_random_images_on_canvases(n_images):
    """
    Show n_images of MNIST placed randomly on individual 1200x1200 canvases.
    """
    for _ in range(n_images):
        # Randomly choose an image
        idx = np.random.randint(0, X_train.shape[0])
        image = X_train[idx]

        # Create a canvas with the image placed randomly
        canvas = place_image_on_canvas(image, width=128, height=128)

        # Display the canvas
        plt.figure(figsize=(5, 5))
        plt.imshow(canvas, cmap='gray')
        plt.axis('off')
        plt.show()


def create_dataset(list_img, list_labels, n_images=100):
    """
    Create a dataset of n_images of MNIST placed randomly on individual 1200x1200 canvases.
    """
    X = []
    #y = []
    coords_x = []
    coords_y = []
    labels = []
    for i in range(n_images):
        image = list_img[i]
        # Create a canvas with the image placed randomly
        canvas, (x, y) = place_image_on_canvas(image, width=128, height=128)
        canvas_norm = canvas / 255
        #flat_canvas = canvas_norm.flatten()
        # Add the canvas to the dataset
        X.append(canvas_norm)
        labels.append(list_labels[i])
        coords_x.append(x)
        coords_y.append(y)
    
    X = np.array(X)
    coords = np.array([coords_x, coords_y]).T
    #y = np.array(y)

    return X, coords, labels

X_train_canvas, coords, y_train = create_dataset(X_train, y_train, n_images=X_train.shape[0])
X_test_canvas, coords_test, y_test = create_dataset(X_test, y_test, n_images=X_test.shape[0])

y_train_onehot = utils.to_categorical(y_train, num_classes=10)
y_test_onehot = utils.to_categorical(y_test, num_classes=10)

np.save('X_train_canvas.npy', X_train_canvas)
np.save('coords.npy', coords)
np.save('y_train.npy', y_train_onehot)

np.save('X_test_canvas.npy', X_test_canvas)
np.save('coords_test.npy', coords_test)
np.save('y_test.npy', y_test_onehot)

#plt.imshow(X_train_canvas[23].reshape(128, 128))
#plt.show()
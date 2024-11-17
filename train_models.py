import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mymodels

class models:
    def __init__(self):
        self.main_folder = '../'
        self.dataset_folder = self.main_folder + 'dataset2/'
        self.checkpoints_folder = self.main_folder + 'checkpoints_sect2/'
        self.logs_folder = self.main_folder + 'logs/'

    def initialise_data_and_model(self):
        self.X = np.load(self.dataset_folder + '/X_cropped.npy')
        self.y = np.load(self.dataset_folder + '/y_train.npy')

        self.X_test = np.load(self.dataset_folder + '/X_cropped.npy')
        self.y_test = np.load(self.dataset_folder + '/y_train.npy')

        self.model = mymodels.sect2()
        self.model.compile()

    def train(self, params=None):
        if params == None:
            params = {'epochs': 30, 
                    'batch_size': 1024, 
                    'tensorboard': True, 
                    'cp_callback': True}
        self.run = self.model.train(self.X, self.y, self.X_test, self.y_test, params, self.logs_folder, self.checkpoints_folder)

    def plot(self):

        history_model = self.run.history
        print("The history has the following data: ", history_model.keys())

        # Plotting the training and validation accuracy during the training
        sns.lineplot(
            x=self.run.epoch, y=history_model[list(history_model)[0]], color="blue", label="Training set"
        )
        sns.lineplot(
            x=self.run.epoch,
            y=history_model[list(history_model)[2]],
            color="red",
            label="Valdation set",
        )
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.show()
        
    def eval_random(self):
        random_sample = np.random.randint(0, self.X_test.shape[0])

        image = self.X_test[random_sample]
        actual = self.y_test[random_sample]
        prediction = self.model.predict(self.X_test[random_sample].reshape(1, 42, 42, 1))

        plt.imshow(image, cmap='gray')
        plt.title(f"Predicted: {prediction.argmax()}")
        plt.show()

class sect2(models):
    def __init__(self):
        super().__init__()

class sect1(models):
    def __init__(self):
        super().__init__()
        self.dataset_folder = self.main_folder + 'dataset/'
        self.checkpoints_folder = self.main_folder + 'checkpoints/'

    def initialise_data_and_model(self):
        self.X = np.load(self.dataset_folder + '/X_train_canvas.npy')
        self.y = np.load(self.dataset_folder + '/coords.npy')

        self.X_test = np.load(self.dataset_folder + '/X_test_canvas.npy')
        self.y_test = np.load(self.dataset_folder + '/coords_test.npy')

        self.model = mymodels.sect1()
        self.model.compile()

class single(models):
    def __init__(self):
        super().__init__()
        self.dataset_folder = self.main_folder + 'dataset/'
        self.checkpoints_folder = self.main_folder + 'checkpoints/'

    def initialise_data_and_model(self):
        self.X = np.load(self.dataset_folder + '/X_train_canvas.npy')
        self.y = np.load(self.dataset_folder + '/y_train.npy')

        self.X_test = np.load(self.dataset_folder + '/X_test_canvas.npy')
        self.y_test = np.load(self.dataset_folder + '/y_test.npy')

        self.model = mymodels.single()
        self.model.compile()
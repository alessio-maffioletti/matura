import random
import matplotlib.pyplot as plt
from tensorflow.keras.backend import clear_session
import optuna
import json
from constants import *


class Optimizer:
    def reduce_model_size(self, conv_layers, dense_layers, factor=0.8, rand_factor=None):
        if rand_factor:
            conv_layers = [max(1, int(layer * random.uniform(rand_factor[0], rand_factor[1]))) for layer in conv_layers if layer > 1]
            dense_layers = [max(1, int(layer * random.uniform(rand_factor[0], rand_factor[1]))) for layer in dense_layers if layer > 1]
            
        else:
            conv_layers = [max(1, int(layer * factor)) for layer in conv_layers if layer > 1]
            dense_layers = [max(1, int(layer * factor)) for layer in dense_layers if layer > 1]

        conv_layers = [layer for layer in conv_layers if layer >= 2]
        dense_layers = [layer for layer in dense_layers if layer >= 2]

        return conv_layers, dense_layers


    def plot_results(self, trainable_params_list, training_time_list, iteration_list, figsize=(15, 5)):
        # Plot the first graph: Trainable parameters vs. Training time
        plt.figure(figsize=figsize)

        plt.subplot(1, 3, 1)  # Plot in the first subplot (1 row, 3 columns)
        plt.plot(trainable_params_list[::-1], training_time_list, marker='o', linestyle='-', color='b')
        plt.xlabel('Trainable Parameters')
        plt.ylabel('Training Time (seconds)')
        plt.title('Trainable Parameters vs Training Time')

        # Plot the second graph: Iteration vs. Training time
        plt.subplot(1, 3, 2)  # Plot in the second subplot
        plt.plot(iteration_list, training_time_list, marker='x', linestyle='-', color='r')
        plt.xlabel('Iteration')
        plt.ylabel('Training Time (seconds)')
        plt.title('Iteration vs Training Time')

        # Plot the third graph: Trainable Parameters vs Iteration
        plt.subplot(1, 3, 3)  # Plot in the third subplot
        plt.plot(iteration_list, trainable_params_list, marker='^', linestyle='-', color='g')
        plt.xlabel('Iteration')
        plt.ylabel('Trainable Parameters')
        plt.title('Trainable Parameters vs Iteration')

        # Show all plots
        plt.tight_layout()
        plt.show()


    def optimize(self, model, conv_layers, dense_layers, target, max_iter=10, layer_factor=0.8, random_factor=None, max_failed_trains=2):
        trainable_params_list = []
        training_time_list = []
        iteration_list = []

        best_conv = conv_layers
        best_dense = dense_layers
        iter_num = 0

        failed_trains = 0

        for i in range(max_iter):
            print(f"Iteration: {i}/{max_iter}")
            print(f"Training model with: {conv_layers}, {dense_layers}")

            #model = sect1()
            trainable_params = model.initialise_data_and_model(conv_layers=conv_layers, dense_layers=dense_layers)
            params = {'epochs': 30,
                    'tensorboard': True, 
                    'cp_callback': False,
                    'weights': None,
                    'stop_at': target,
                    'save_final': True,
                    'weight_string': f'_optimize_{i}'
                    }

        #try:
            reached_target, training_time = model.train(params)

            print(f"Training time: {training_time}s")
        
            trainable_params_list.append(trainable_params)
            training_time_list.append(training_time)
            iteration_list.append(i)

            model.plot()
            #model.eval_random()

            # If the target is not reached, break the loop
            if not reached_target:
                failed_trains += 1
                print(f"Target value not reached, attempt: {failed_trains}")
                if failed_trains >= max_failed_trains:
                    print("Target value not reached, done optimizing")
                    break
            else:
                failed_trains = 0

            if training_time == min(training_time_list):                
                best_conv = conv_layers
                best_dense = dense_layers
                iter_num = i
                
            # Reduce the model size for the next iteration
            conv_layers, dense_layers = self.reduce_model_size(conv_layers, dense_layers, factor=layer_factor, rand_factor=random_factor)

            #except Exception as e:
                #print(f"Error during training on iteration {i}: {str(e)}")
                #break  # Exit the loop if training fails due to an error

            clear_session()

        self.plot_results(trainable_params_list, training_time_list, iteration_list)
        return best_conv, best_dense, iter_num
    

class BayesianOptimizer:
    def __init__(self):
        self.study = optuna.create_study(direction="minimize")
        self.best_time = 600

    def _set_initial_params(self, initial_params):
        # Define default parameters
        default_params = {
            'min_conv_layers': 1,
            'max_conv_layers': 5,
            'min_dense_layers': 1,
            'max_dense_layers': 5,
            'min_conv_size': 4,
            'max_conv_size': 256,
            'min_dense_size': 4,
            'max_dense_size': 256,
            'epochs': 10,
            'flatten_type': ['global_average', 'flatten'],
            'activation': ['relu', 'sigmoid'],
            'optimizer': ['adam', 'sgd'],
            'learning_rate': [0.001, 0.01],
            'dropout': [0.0, 0.1],
        }

        # If initial_params is None, use the default parameters
        if initial_params is None:
            initial_params = default_params
        else:
            # Otherwise, fill missing keys with the default value
            for key, value in default_params.items():
                initial_params.setdefault(key, value)

        return initial_params
    
    def _convert_to_params(self, trial_params, conv_layers, dense_layers):
        trial_params['conv_layers'] = conv_layers
        trial_params['dense_layers'] = dense_layers

        return trial_params
    def write(self, trial_params, training_time):
        # Create a dictionary for the new trial
        trial_data = {
            "training_time": training_time,
            "params": trial_params
        }
        
        # Try to read the existing file, or create an empty list if it doesn't exist
        try:
            with open(OPTIMIZER_FOLDER + "trials.json", "r") as file:
                trials = json.load(file)
                file.close()
        except (FileNotFoundError, json.JSONDecodeError):
            trials = []

        # Append the new trial data
        trials.append(trial_data)

        # Sort trials by training time (ascending order)
        trials_sorted = sorted(trials, key=lambda x: x['training_time'])

        # Write the sorted trials back to the JSON file
        with open(OPTIMIZER_FOLDER + "trials.json", "w") as file:
            json.dump(trials_sorted, file, indent=4)
            file.close()

    def optimize(self, model, target, initial_params=None, max_trials=20, write_to_file=False):
        self.study.optimize(lambda trial: self.optimize_model(trial, model, target, initial_params, write_to_file), n_trials=max_trials)
        print("Best trial:", self.study.best_trial.params)

        return self.study.best_trial



    def optimize_model(self, trial, model, target, initial_params, write_to_file):
        initial_params = self._set_initial_params(initial_params)

        num_conv_layers = trial.suggest_int("num_conv_layers", initial_params['min_conv_layers'], initial_params['max_conv_layers'])
        num_dense_layers = trial.suggest_int("num_dense_layers", initial_params['min_dense_layers'], initial_params['max_dense_layers'])

        conv_layers = [trial.suggest_int(f"conv_{i}_size", initial_params['min_conv_size'], initial_params['max_conv_size']) for i in range(num_conv_layers)]
        dense_layers = [trial.suggest_int(f"dense_{i}_size", initial_params['min_dense_size'], initial_params['max_dense_size']) for i in range(num_dense_layers)]

        flatten_type = trial.suggest_categorical("flatten_type", initial_params['flatten_type'])
        activation = trial.suggest_categorical("activation", initial_params['activation'])
        optimizer = trial.suggest_categorical("optimizer", initial_params['optimizer'])
        learning_rate = trial.suggest_float("learning_rate", initial_params['learning_rate'][0], initial_params['learning_rate'][1])
        dropout = trial.suggest_float("dropout", initial_params['dropout'][0], initial_params['dropout'][1])

        
        suggested_params = self._convert_to_params(trial.params, conv_layers, dense_layers)
        

        model.initialise_data_and_model(suggested_params)
        params = {'epochs': initial_params['epochs'],
                'stop_at': target,
                'max_time': self.best_time,
                }
        reached_target, training_time = model.train(params)

        self.best_time = min(self.best_time, training_time)

        if write_to_file:
            print("writing to file")
            self.write(trial.params, training_time)

        if reached_target:
            return training_time
        else:
            return float("inf")

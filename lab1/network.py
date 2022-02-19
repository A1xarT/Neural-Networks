import math
import random
import pickle
import statistics
import numpy as np


def activation_function(x):
    return 1.0 / (1.0 + math.exp(-x))


def transfer_derivative(y):
    return y * (1.0 - y)


class Network:
    def __init__(self):
        self.input_layer_size = 100 * 100
        self.output_layer_size = 20 * 2
        self.hidden_layer_size = 100
        self.hidden_neurons = []
        self.output_neurons = []
        self.weights_input = []
        self.weights_output = []

    def set_start_weights(self):
        start_weights_rate = 10.0 / self.input_layer_size
        for i in range(self.input_layer_size):
            self.weights_input.append([])
            for j in range(self.hidden_layer_size):
                self.weights_input[i].append(start_weights_rate * random.random())
        for i in range(self.hidden_layer_size):
            self.weights_output.append([])
            for j in range(self.output_layer_size):
                self.weights_output[i].append(start_weights_rate * random.random())
        print(123)

    def load_weights(self):
        weights = open('weights_1.pickle', 'rb')
        self.weights_input = pickle.load(weights)
        weights = open('weights_2.pickle', 'rb')
        self.weights_output = pickle.load(weights)
        weights.close()

    def save_weights(self):
        weights = open('weights_1.pickle', 'wb')
        pickle.dump(self.weights_input, weights)
        weights = open('weights_2.pickle', 'wb')
        pickle.dump(self.weights_output, weights)
        weights.close()

    def calculate_weights(self, input_neurons):
        self.hidden_neurons.clear()
        self.output_neurons.clear()
        for i in range(self.hidden_layer_size):  # calculating weights between input and hidden layers
            x = 0.0
            for j in range(self.input_layer_size):
                x += input_neurons[j] * self.weights_input[j][i]
            y = activation_function(x)
            self.hidden_neurons.append(y)
        for i in range(self.output_layer_size):  # calculating weights between hidden and output layers
            x = 0.0
            for j in range(self.hidden_layer_size):
                x += self.hidden_neurons[j] * self.weights_output[j][i]
            y = activation_function(x)
            self.output_neurons.append(y)

    def train_accuracy(self, input_neurons, expected_values, accuracy):
        max_error = 1.0
        epoch_counter = 0
        while max_error > accuracy:
            self.calculate_weights(input_neurons)  # calculating weights and activated values
            hidden_layer_deltas = []  # delta values for output neurons
            output_layer_deltas = []  # delta values for hidden neurons
            errors = []
            for i in range(self.output_layer_size):
                errors.append((self.output_neurons[i] - expected_values[i]) ** 2)
                output_layer_deltas.append(
                    (self.output_neurons[i] - expected_values[i]) * transfer_derivative(self.output_neurons[i]))
            for i in range(self.hidden_layer_size):
                error = 0.0
                for j in range(self.output_layer_size):
                    error += self.weights_output[i][j] * output_layer_deltas[j]
                hidden_layer_deltas.append(error * transfer_derivative(self.hidden_neurons[i]))
            self.update_weights(input_neurons, hidden_layer_deltas, output_layer_deltas, 0.2)
            max_error = np.max(errors)
            epoch_counter += 1
            # print(errors)
            print(epoch_counter)
        return epoch_counter

    def train(self, input_neurons, expected_values, epoch_number):
        for n in range(epoch_number):
            self.calculate_weights(input_neurons)  # calculating weights and activated values
            hidden_layer_deltas = []  # delta values for output neurons
            output_layer_deltas = []  # delta values for hidden neurons
            errors = []
            for i in range(self.output_layer_size):
                errors.append((self.output_neurons[i] - expected_values[i]) ** 2)
                output_layer_deltas.append(
                    (self.output_neurons[i] - expected_values[i]) * transfer_derivative(self.output_neurons[i]))
            for i in range(self.hidden_layer_size):
                error = 0.0
                for j in range(self.output_layer_size):
                    error += self.weights_output[i][j] * output_layer_deltas[j]
                hidden_layer_deltas.append(error * transfer_derivative(self.hidden_neurons[i]))
            self.update_weights(input_neurons, hidden_layer_deltas, output_layer_deltas, 0.01)

    def update_weights(self, input_neurons, hidden_deltas, output_deltas, l_rate):
        for i in range(self.input_layer_size):
            for j in range(self.hidden_layer_size):
                self.weights_input[i][j] -= l_rate * hidden_deltas[j] * input_neurons[i]
        for i in range(self.hidden_layer_size):
            for j in range(self.output_layer_size):
                self.weights_output[i][j] -= l_rate * output_deltas[j] * self.hidden_neurons[i]

    def get_answer(self, input_neurons):
        self.calculate_weights(input_neurons)
        return self.output_neurons

    def get_error(self, input_neurons, expected_output):
        self.calculate_weights(input_neurons)
        errors = []
        for i in range(len(expected_output)):
            errors.append((self.output_neurons[i] - expected_output[i]) ** 2)
        print(np.max(errors))
        return np.max(errors)

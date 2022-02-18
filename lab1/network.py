import math
import random
import pickle


def activation_function(x):
    return 1 / (1 + math.pow(math.e, -x))


class Network:
    def __init__(self):
        self.input_layer_size = 100 * 100
        self.output_layer_size = 40 * 2
        self.hidden_layer_size = 5
        self.hidden_neurons = []
        self.output_neurons = []
        self.weights_input = []
        self.weights_output = []

    def set_start_weights(self):
        for i in range(self.input_layer_size):
            self.weights_input.append([])
            for j in range(self.hidden_layer_size):
                self.weights_input[i].append(random.random())
        for i in range(self.hidden_layer_size):
            self.weights_output.append([])
            for j in range(self.output_layer_size):
                self.weights_output[i].append(random.random())

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
        self.hidden_neurons = []
        self.output_neurons = []
        for i in range(self.hidden_layer_size):  # calculating weights between input and hidden layers
            x = 0
            for j in range(self.input_layer_size):
                x += input_neurons[j] * self.weights_input[j][i]
            y = activation_function(x)
            self.hidden_neurons.append(y)
        for i in range(self.output_layer_size):  # calculating weights between hidden and output layers
            x = 0
            for j in range(self.hidden_layer_size):
                x += self.hidden_neurons[j] * self.weights_output[j][i]
            y = activation_function(x)
            self.output_neurons.append(y)

    def train(self, input_neurons, output_values, epoch_number):
        for n in range(epoch_number):
            self.calculate_weights(input_neurons)  # calculating weights and activated values
            d3_arr = []
            prev_output_weights = self.weights_output
            for i in range(self.output_layer_size):  # correcting weights between hidden and output layers
                d3 = (output_values[i] - self.output_neurons[i]) * self.output_neurons[i] * (1 - self.output_neurons[i])
                d3_arr.append(d3)
                for j in range(self.hidden_layer_size):
                    dw = d3 * self.hidden_neurons[j]
                    self.weights_output[j][i] += dw
            d2_arr = []
            for i in range(self.hidden_layer_size):  # calculating d2
                d_sum = 0
                for j in range(self.output_layer_size):
                    d_sum += d3_arr[j] * prev_output_weights[i][j]
                d = d_sum / self.output_layer_size * self.hidden_neurons[i] * (1 - self.hidden_neurons[i])
                d2_arr.append(d)
            for i in range(self.hidden_layer_size):  # correcting weights between input and hidden layers
                for j in range(self.input_layer_size):
                    dw = d2_arr[i] * input_neurons[j]
                    self.weights_input[j][i] += dw

    def get_answer(self, input_neurons):
        self.calculate_weights(input_neurons)
        return self.output_neurons

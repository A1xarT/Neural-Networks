import math


def activation_function(x):
    return 1 / (1 + math.pow(math.e, -x))


def x_sum(x_values, weights):
    res = 0
    for i in range(len(x_values)):
        res += x_values[i] * weights[i]
    return res


class Network:
    def __init__(self):
        self.input_layer_size = 100 * 100
        self.output_layer_size = 40 * 2
        self.hidden_layer_size = self.input_layer_size + self.output_layer_size

    def train(self):
        print()

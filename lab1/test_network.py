import math

# def train_network(nw):
#     x_values = [0, 1]
#     yR = 1
#     for k in range(0, 10000):
#         yM = activation_function(x_sum(x_values, nw.weights))
#         di = yM * (1 - yM) * (yR - yM)
#         for i in range(0, len(x_values)):
#             dw = x_values[i] * di
#             nw.weights[i] = nw.weights[i] + dw
#     return nw

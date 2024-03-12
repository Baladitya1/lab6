import numpy as np
import matplotlib.pyplot as plt
from math import exp


def sigmoid_activation(training_data, training_output, weights, alpha, epochs):
    errors = []
    epoch_num = 0
    for epoch in range(epochs):
        error_sum = 0
        epoch_num += 1
        for i in range(len(training_data)):
            x = training_data[i]
            y = training_output[i]
            pred = np.dot(weights, x)
            output = 1 / (1 + exp(-pred))
            error = y - output  # Corrected error calculation
            for j in range(len(weights)):
                weights[j] += alpha * error * x[j]
            error_sum += error ** 2
        errors.append(error_sum)
        if error_sum <= 0.002:
            return weights, errors, epoch_num
    return weights, errors, epoch_num

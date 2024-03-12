import numpy as np
import matplotlib.pyplot as plt
from math import exp

def step_activation(training_data, weights, learning_rate, epochs):
    errors = []  # List to store the sum of squared errors for each epoch
    epoch_num = 0  # Initialize the epoch number counter
    for epoch in range(epochs):
        error_sum = 0  # Initialize the sum of squared errors for the current epoch
        epoch_num += 1  # Increment the epoch number counter
        for x in training_data:  # Loop through each training data point
            pred = weights[0] * x[0] + weights[1] * x[1] + weights[2] * 1  # Compute the prediction
            output = pred > 0  # Apply step activation function
            error = x[2] - int(output)  # Compute the error
            weights[0] += learning_rate * error * x[0]  # Update weight for feature 1
            weights[1] += learning_rate * error * x[1]  # Update weight for feature 2
            weights[2] += learning_rate * error * 1  # Update bias weight
            error_sum += error ** 2  # Add the squared error to the sum of squared errors
        errors.append(error_sum)  # Append the sum of squared errors for the current epoch
        if error_sum <= 0.002:  # Check if convergence criteria is met
            return weights, errors, epoch_num  # Return weights, errors, and epoch number
    return weights, errors, epoch_num  # Return weights, errors, and epoch number if maximum epochs reached


def sigmoid_activation(training_data, weights, learning_rate, epochs):
    errors = []
    epoch_num = 0
    for epoch in range(epochs):
        error_sum = 0
        epoch_num += 1
        for x in training_data:
            pred = weights[0] * x[0] + weights[1] * x[1] + weights[2] * 1
            output = 1 / (1 + exp(-pred))
            error = x[2] - output
            weights[0] += learning_rate * error * x[0]
            weights[1] += learning_rate * error * x[1]
            weights[2] += learning_rate * error * 1
            error_sum += error ** 2
        errors.append(error_sum)
        if error_sum <= 0.002:
            return weights, errors, epoch_num
    return weights, errors, epoch_num


def bi_polar_activation(training_data, weights, learning_rate, epochs):
    errors = []  # List to store the sum of squared errors for each epoch
    epoch_num = 0  # Initialize the epoch number counter
    for epoch in range(epochs):
        error_sum = 0  # Initialize the sum of squared errors for the current epoch
        epoch_num += 1  # Increment the epoch number counter
        for x in training_data:  # Loop through each training data point
            pred = weights[0] * x[0] + weights[1] * x[1] + weights[2] * 1  # Compute the prediction
            if pred > 0:  # Apply bipolar activation function
                output = 1
            elif pred == 0:
                output = 0
            else:
                output = -1
            error = x[2] - output  # Compute the error
            weights[0] += learning_rate * error * x[0]  # Update weight for feature 1
            weights[1] += learning_rate * error * x[1]  # Update weight for feature 2
            weights[2] += learning_rate * error * 1  # Update bias weight
            error_sum += error ** 2  # Add the squared error to the sum of squared errors
        errors.append(error_sum)  # Append the sum of squared errors for the current epoch
        if error_sum <= 0.002:  # Check if convergence criteria is met
            return weights, errors, epoch_num  # Return weights, errors, and epoch number
    return weights, errors, epoch_num  # Return weights, errors, and epoch number if maximum epochs reached


def reLU_activation(training_data, weights, learning_rate, epochs):
    errors = []
    epoch_num = 0
    for epoch in range(epochs):
        error_sum = 0
        epoch_num += 1
        for x in training_data:
            pred = weights[0] * x[0] + weights[1] * x[1] + weights[2] * 1
            if pred > 0:
                output = pred
            else:
                output = 0
            error = x[2] - output
            weights[0] += learning_rate * error * x[0]
            weights[1] += learning_rate * error * x[1]
            weights[2] += learning_rate * error * 1
            error_sum += error ** 2
        errors.append(error_sum)
        if error_sum <= 0.002:
            return weights, errors, epoch_num  
    return weights, errors, epoch_num 


def test_perceptron(weights, test_data, activation_function):
    correct_predictions = 0
    total_predictions = len(test_data)

    for x in test_data:
        pred = weights[0] * x[0] + weights[1] * x[1] + weights[2] * 1
        if activation_function == 'step':
            output = int(pred > 0)
        elif activation_function == 'sigmoid':
            output = 1 / (1 + exp(-pred))
        elif activation_function == 'bi_polar':
            if pred > 0:
                output = 1
            elif pred == 0:
                output = 0
            else:
                output = -1
        elif activation_function == 'reLU':
            output = max(0, pred)

        if round(output) == x[2]:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy


training_data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
weights = [10, 0.2, -0.75]
learning_rate = 0.05
epochs = 1000

activation_functions = ['step', 'sigmoid', 'bi_polar', 'reLU']
activated_weights = [SWeights, SiWeights, Bweights, RWeights]

for activation_function, activated_weight in zip(activation_functions, activated_weights):
    accuracy = test_perceptron(activated_weight, training_data, activation_function)
    print(f"Activation function: {activation_function}, Accuracy: {accuracy * 100}")


SWeights, SErrors, SEpoch_num = step_activation(training_data, weights, learning_rate, epochs)

# Plotting
print("Step_activation")
print(f"Finalized weights are W0:{SWeights[0]}, W1:{SWeights[1]},W2:{SWeights[2]}")
print(f"number of Epoches needed is {SEpoch_num}")
plt.plot(range(len(SErrors)), SErrors)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('step activation')
plt.show()

Bweights, Berrors, Bepoch_num = bi_polar_activation(training_data, weights, learning_rate, epochs)

# Plotting
print("Bi-polar activation")
print(f"Finalized weights are W0:{Bweights[0]}, W1:{Bweights[1]},W2:{Bweights[2]}")
print(f"number of Epoches needed is {Bepoch_num}")
plt.plot(range(len(Berrors)), Berrors)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('Bi-polar activation')
plt.show()

SiWeights, SiErrors, SiEpoch_num = sigmoid_activation(training_data, weights, learning_rate, epochs)

# Plotting
print("Sigmoid activation")
print(f"Finalized weights are W0:{SiWeights[0]}, W1:{SiWeights[1]},W2:{SiWeights[2]}")
print(f"number of Epoches needed is {SiEpoch_num}")
plt.plot(range(len(SiErrors)), SiErrors)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('sigmoid activation')
plt.show()

RWeights, RErrors, REpoch_num = reLU_activation(training_data, weights, learning_rate, epochs)

# Plotting
print("reLU activation")
print(f"Finalized weights are W0:{RWeights[0]}, W1:{RWeights[1]},W2:{RWeights[2]}")
print(f"number of Epoches needed is {REpoch_num}")
plt.plot(range(len(RErrors)), RErrors)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('reLU activation')
plt.show()

# question 3
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
relu_iter = []
bi_iter = []
si_iter = []
st_iter = []

for rate in learning_rates:
    _, _, rlepoch_num = reLU_activation(training_data, weights, rate, epochs)
    _, _, SiEpoch_num = sigmoid_activation(training_data, weights, rate, epochs)
    _, _, Bepoch_num = bi_polar_activation(training_data, weights, rate, epochs)
    _, _, SEpoch_num = step_activation(training_data, weights, rate, epochs)
    relu_iter.append(rlepoch_num)
    si_iter.append(SiEpoch_num)
    bi_iter.append(Bepoch_num)
    st_iter.append(SEpoch_num)

plt.plot(learning_rates, relu_iter)
plt.title('reLU activation')
plt.xlabel('Learning rate')
plt.ylabel('Epochs')
plt.show()

plt.plot(learning_rates, st_iter)
plt.title('Step activation activation')
plt.xlabel('Learning rate')
plt.ylabel('Epochs')
plt.show()

plt.plot(learning_rates, si_iter)
plt.title('Sigmoid activation')
plt.xlabel('Learning rate')
plt.ylabel('Epochs')
plt.show()

plt.plot(learning_rates, bi_iter)
plt.title('Bi-polar activation')
plt.xlabel('Learning rate')
plt.ylabel('Epochs')
plt.show()

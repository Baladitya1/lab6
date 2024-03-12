from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Define training data and output for XOR gate
training_data_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_xor = np.array([0, 1, 1, 0])

# Define training data and output for AND gate
training_data_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_and = np.array([0, 0, 0, 1])

# Learning rate (alpha)
learning_rate = 0.05

# Initialize MLP classifier for XOR gate
clf_xor = MLPClassifier(solver='sgd', learning_rate_init=learning_rate, hidden_layer_sizes=(4,), random_state=1)
# Fit the model for XOR gate
clf_xor.fit(training_data_xor, output_xor)
# Predict outputs for XOR gate
X = clf_xor.predict([[0, 1], [1, 1], [0, 0], [1, 0]])

# Initialize MLP classifier for AND gate
clf_and = MLPClassifier(solver='sgd', learning_rate_init=learning_rate, hidden_layer_sizes=(5,), random_state=1)
# Fit the model for AND gate
clf_and.fit(training_data_and, output_and)
# Predict outputs for AND gate
Y = clf_and.predict([[0, 1], [1, 1], [0, 0], [1, 0]])

# Print outputs for XOR and AND gates
print(f"Output for XOR gate is : {X}")
print(f"Output for AND gate is : {Y}")

# Load dataset from file
data = np.loadtxt("kinematic_features.txt")
X = data
# Define output labels (0 for one class, 1 for another)
y = np.concatenate((np.zeros(41), np.ones(55)))

# Split the data into training and testing sets randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=41)

# Initialize MLP classifier
clf = MLPClassifier(solver='sgd', learning_rate_init=learning_rate, random_state=1)
# Fit the model
clf.fit(X_train, y_train)
# Predict outputs
y_pred = clf.predict(X_test)

# Print accuracy score of the classifier
print(f"Accuracy score: {clf.score(X_test, y_test)}")

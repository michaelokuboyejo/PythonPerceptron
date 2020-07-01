import numpy as np


class NeuralNetwork:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        np.random.seed(1)
        self.weights = 2 * np.random.random((num_classes, 1)) - 1

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def train(self, training_input, training_output, num_iterations=1000):
        assert training_input.shape[1] == self.num_classes
        assert training_input.shape[0] == training_output.shape[0]
        for iteration in range(num_iterations):
            input_layer = training_input
            output = self.sigmoid(np.dot(input_layer, self.weights))
            error = training_output - output
            adjustment = error * self.sigmoid_derivative(output)
            self.weights += np.dot(input_layer.T, adjustment)

    def predict(self, test_input):
        return self.sigmoid(np.dot(test_input, self.weights))


if __name__ == '__main__':
    training_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_outputs = np.array([[0, 1, 1, 0]]).T
    nn = NeuralNetwork(3)
    nn.train(training_inputs, training_outputs, 100000)
    prediction = nn.predict(np.array([0, 0, 1]))
    print(prediction)

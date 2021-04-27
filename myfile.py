import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((2, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):

            output = self.think(training_inputs)

            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

                                # синие
    training_inputs = np.array([[15.42, 52.808],
                                [18.23, 89.094],
                                [31.811, 125.34],
                                [70.916, 131.734],
                                [83.184, 116.567],
                                [43.226, 115.12],
                                [58.789, 106.071],
                                [80.712, 95.493],
                                [45.444, 87.531],
                                [30.614, 73.599],
                                [65.472, 80.337],
                                [47.266, 55.153],
                                [65.062, 58.221],
                                [25.255, 40.114],
                                [44.456, 37.476],
                                #красные
                                [43.111, 15.264],
                                [44.49, 6.21],
                                [52.886, 22.2],
                                [87.424, 38.651],
                                [105.611, 34.341],
                                [104.137, 13.953],
                                [120.853, 52.876],
                                [135.096, 36.016],
                                [151.232, 60.762],
                                [151.232, 34.417],
                                [166.833, 45.851],
                                [165.516, 23.102],
                                [73.492, 13.854]])

    for i in range(28):
        training_inputs[i][0] = training_inputs[i][0] * 0.001
        training_inputs[i][1] = training_inputs[i][1] * 0.001

    training_outputs = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).T

    neural_network.train(training_inputs, training_outputs, 10000)

    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    for i in range(15):
        plt.scatter(training_inputs[i][0], training_inputs[i][1], s=10, c='blue')
    for i in range(13):
        plt.scatter(training_inputs[i+15][0], training_inputs[i+15][1], s=10, c='red')
    plt.grid(True)
    plt.show()

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))

    print("New situation: input data = ", A, B)
    print("Output data: ")
    if np.around(neural_network.think(np.array([A, B]))) == 1:
        print("RED")
    else:
        print("BLUE")



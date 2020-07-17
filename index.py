import numpy as np
import pandas as pd


class Layer:
    def __init__(self, size, previous_layer_size):
        self._size = size
        self._bias = np.random.randn(size, 1)
        self._weigths = np.random.randn(size, previous_layer_size)
        self.nabla_b = np.zeros(self._bias.shape)
        self.nabla_w = np.zeros(self._weigths.shape)

    def feedforward(self, activations):
        assert self._weigths.shape[1] == activations.shape[
            0], "Wrong dimension of activation layer. {} (dim 0) need, {} (dim 0) given".format(self._weigths.shape[1],
                                                                                                activations.shape[0])
        return sigmoid(np.dot(self._weigths, activations)) # TODO: return +bias

    def backpropagation(self, activation, e, previous_activation):
        delta = e * activation * (1 - activation)
        self.nabla_b += delta
        self.nabla_w += np.dot(delta, np.transpose(previous_activation))
        return delta

    @property
    def bias(self):
        return self._bias

    @property
    def weights(self):
        return self._weigths

    def discard_nablas(self):
        self.nabla_b = np.zeros(self._bias.shape)
        self.nabla_w = np.zeros(self._weigths.shape)

    def update_weight(self, learning_rate, number_of_iterations):
        delta = (learning_rate / number_of_iterations) * self.nabla_w
        self._weigths -= delta

    def update_bias(self, learning_rate, number_of_iterations):
        self._bias -= (learning_rate / number_of_iterations) * self.nabla_b


class Net:
    def __init__(self, sizes):
        self._sizes = sizes
        self._layers = []
        for size, prev_layer_size in zip(sizes[1:], sizes[:-1]):
            self._layers.append(Layer(size, prev_layer_size))

    def feedforward(self, activation):
        for layer in self._layers:
            activation = layer.feedforward(activation)
        return activation

    # for b, w in zip(self.biases, self.weights):
    #     layer = np.dot(w, layer)
    #     layer = layer + b
    #     layer = sigmoid(layer)
    # return layer

    def train(self, training_data, epochs, mini_bucket_size, learning_rate, test_data=None):
        training_data = training_data.values
        test_data = test_data
        for j in range(epochs):
            # np.random.shuffle(training_data)
            mini_buckets = [
                training_data[k:k + mini_bucket_size]
                for k in range(0, len(training_data), mini_bucket_size)]
            for mini_bucket in mini_buckets:
                self.update_mini_bucket(mini_bucket, learning_rate)
            # if test_data:
            print("Epoch {} : {}%".format(j, self.test(test_data) * 100))
            # else:
            # print("Epoch {} complete".format(j))

    def update_mini_bucket(self, mini_bucket, learning_rate):
        for layer in self._layers:
            layer.discard_nablas()
        for data in mini_bucket:
            y, x = data[0], np.resize(data[1:] / 256, (len(data) - 1, 1))
            self.backprop(x, y)
        for layer in self._layers:
            layer.update_weight(learning_rate, len(mini_bucket))
            layer.update_bias(learning_rate, len(mini_bucket))

    def test(self, test_data):
        number_of_tests = len(test_data.values)
        correct_tests = 0
        for test in test_data.values:
            correct_answer, test = test[0], test[1:] / 256
            result = self.feedforward(test)
            if np.argmax(result) == correct_answer:
                correct_tests += 1
        return correct_tests / number_of_tests

    def backprop(self, input_layer, correct_answer):
        # feedforward
        activation = input_layer
        activations = [activation]
        for layer in self._layers:
            activations.append(layer.feedforward(activations[-1]))
        # backward
        correct_vector = np.zeros((self._sizes[-1], 1))
        correct_vector[int(correct_answer)] = 1
        e = cost_derivative(activations[-1], correct_vector)
        for layer, activation, prev_activation in zip(self._layers[::-1], activations[::-1], activations[-2::-1]):
            delta = layer.backpropagation(activation, e, prev_activation)
            e = np.dot(layer.weights.T, e)


        # for l in range(2, len(self.sizes)):
        #     z = zs[-l]
        #     sp = sigmoid_prime(z)
        #     delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
        #     nabla_b[-l] = delta
        #     nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        # return nabla_b, nabla_w


def cost_derivative(activations, correct):
    return 2*(activations - correct)


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def upload_data_from_csv(path):
    images = pd.read_csv(path)
    # images = np.genfromtxt(path, delimiter=',')
    return images


def run_app():
    train_file_path = "mnist_data/mnist_train.csv"
    test_file_path = "mnist_data/mnist_test.csv"
    training_test_file_path = "mnist_data/mnist_train_100.csv"
    train_data = upload_data_from_csv(train_file_path)
    test_data = upload_data_from_csv(test_file_path)
    training_test_data = upload_data_from_csv(training_test_file_path)

    accuracy = 0
    while accuracy < 0.2:
        net = Net([784, 200, 10])

        net.train(training_data=train_data, epochs=10, mini_bucket_size=10, learning_rate=0.125,
                  test_data=training_test_data)

        accuracy = net.test(test_data)
        print("Net accuracy: ", accuracy * 100, "%", sep="")


if __name__ == "__main__":
    run_app()

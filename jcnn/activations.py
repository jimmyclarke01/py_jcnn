import numpy as np


class Activations:
    def __init__(self):

        function = lambda x: 1 / (1 + np.exp(-x))
        derivative = lambda fx: fx * (1 - fx)
        self.logistic = ActivationFunction(function, derivative)

        function = lambda x: 0 if x < 0 else x
        derivative = lambda fx: 0 if fx < 0 else 1
        self.relu = ActivationFunction(function, derivative)

        function = lambda x: np.exp(x) / np.sum(np.exp(x))
        self.softmax = ActivationFunction(function, None)

    @property
    def logistic(self):
        return self._logistic

    @logistic.setter
    def logistic(self, value):
        self._logistic = value

    @property
    def relu(self):
        return self._relu

    @relu.setter
    def relu(self, value):
        self._relu = value

    @property
    def softmax(self):
        return self._softmax

    @softmax.setter
    def softmax(self, value):
        self._softmax = value


class ActivationFunction:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, value):
        self._function = value

    @property
    def derivative(self):
        return self._derivative

    @derivative.setter
    def derivative(self, value):
        self._derivative = value

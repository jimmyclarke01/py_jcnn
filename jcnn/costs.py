import numpy as np


class Costs:
    def __init__(self):

        function = lambda x, y: (1 / 2) * np.sum((np.subtract(x, y)) ** 2)
        derivative = lambda x, y: - np.subtract(x, y)
        self.mean_squared = CostFunction(function, derivative)


    @property
    def mean_squared(self):
        return self._mean_squared

    @mean_squared.setter
    def mean_squared(self, value):
        self._mean_squared = value


class CostFunction:
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

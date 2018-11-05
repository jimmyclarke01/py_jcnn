import numpy as np

class NNLayer:

    def __init__(self, n_inputs, n_outputs, activation):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.x = np.ones(self.n_inputs + 1)

        self.W = self._initialiseWeights()

        self.h = np.zeros(self.n_outputs)
        self.activation = activation
        self._initialiseWeights()

    def _initialiseWeights(self):
        return np.random.normal(loc=0.0, scale=np.sqrt(1/self.n_inputs), size=(self.n_inputs + 1, self.n_outputs))

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, value):
        self._activation = value

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, value):
        self._W = value

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, value):
        self._h = value

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def n_inputs(self):
        return self._n_inputs

    @n_inputs.setter
    def n_inputs(self, value):
        self._n_inputs = value

    @property
    def n_outputs(self):
        return self._n_outputs

    @n_outputs.setter
    def n_outputs(self, value):
        self._n_outputs = value

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, value):
        self._delta = value
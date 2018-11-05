import numpy as np

class NNLayer:

    def __init__(self, n_inputs, n_outputs):
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._inputs = np.zeros(self._n_inputs)
        self._outputs = np.zeros(self._n_outputs)
        self._weights = np.zeros((self._n_inputs+1, self._n_outputs))

    def getInputSize(self):
        return self._n_inputs

    def getOutputSize(self):
        return self._n_outputs

    def getWeights(self):
        return self._weights

    def initialiseWeights(self):
        return 0
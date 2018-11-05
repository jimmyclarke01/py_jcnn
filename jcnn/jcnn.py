import numpy as np


class NN:
    def __init__(self, input_size):
        self._layers = []
        self._output_size = input_size
        self._input_size = input_size

    def addLayer(self, layer):
        if layer.getInputSize() == self._output_size:
            self._layers.append(layer)
            self._output_size = layer.getOutputSize()
        else:
            print("Sizes don't match!")

    def initialiseWeights(self):
        for layer in self._layers:
            layer.initialiseWeights()





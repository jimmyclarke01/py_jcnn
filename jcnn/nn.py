import numpy as np
import jcnn

class NN:
    def __init__(self, n_inputs, activation=None, cost=None):
        self.layers = []
        self.n_outputs = n_inputs
        self.n_inputs = n_inputs
        self.activation = activation
        self.x = np.zeros(self.n_inputs + 1)
        self.h = np.zeros(self.n_inputs + 1)

        if self.activation is None:
            self.activation = jcnn.Activations().logistic

        self.cost = cost

        if self.cost is None:
            self.cost = jcnn.Costs().mean_squared

    def addLayer(self, layer):
        if layer.n_inputs == self.n_outputs:
            self.layers.append(layer)
            self.n_outputs = layer.n_outputs
        else:
            print("Sizes don't match!")

    def initialiseWeights(self):
        for layer in self._layers:
            layer.initialiseWeights()

    def forwardProp(self, input):

        for i, layer in enumerate(self.layers):

            # INPUT LAYER
            if i == 0:
                layer.x[1:] = input
            layer.h = np.matmul(layer.x, layer.W)
            y = np.apply_along_axis(layer.activation.function, 0, layer.h)

            # NOT OUTPUT LAYER
            if i != (self.n_layers - 1):
                self.layers[i+1].x[1:] = y
            else:
                return y

    def backProp(self, output, target):

        # OUTPUT LAYER

        # error = (expected - output) * activation_derivative(h)

        yHat = output
        y = target
        z3 = self.layers[1].h
        delta3 = np.multiply(-(y-yHat), self.activation.derivative(z3))
        delta3 = np.asmatrix(delta3)
        a2 = self.layers[0].x
        a2 = np.asmatrix(a2)

        dJdW2 = np.dot(a2.T, delta3)

        w2 = self.layers[1].W
        w2 = np.asmatrix(w2)

        z2 = self.layers[0].h

        z2 = np.asmatrix(z2)
        derivative = np.apply_along_axis(self.activation.derivative, 0, z2)

        delta2 = np.dot(delta3, w2.T).T * derivative

        x = self.layers[0].x

        dJdW1 = np.dot(x.T, delta2)

        lr = 0.3


        self.layers[0].W += np.multiply(dJdW1, lr)

        self.layers[1].W += np.multiply(dJdW2, lr)

        return


        d3 = np.multiply(self.cost.derivative(output, target),
                         self.activation.derivative(self.layers[1].h))

        dJdW2 = np.outer(self.layers[1].x, d3)

        # HIDDEN LAYER

        # error = (weight k * error j) * activation_derivative(h)

        d2 = np.dot(d3, self.layers[1].W) * np.apply_along_axis(self.activation.derivative, 0, self.layers[1].h)

        dJdW1 = np.dot(self.layers[0].x.T, d2)

        lr = 0.3

        self.layers[0].W += lr * dJdW1

        self.layers[1].W += lr * dJdW2




    @property
    def n_layers(self):
        return len(self.layers)

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, value):
        self._layers = value

    @property
    def n_outputs(self):
        return self._output_size

    @n_outputs.setter
    def n_outputs(self, value):
        self._output_size = value

    @property
    def n_inputs(self):
        return self._input_size

    @n_inputs.setter
    def n_inputs(self, value):
        self._input_size = value

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, value):
        self._activation = value

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, value):
        self._cost = value


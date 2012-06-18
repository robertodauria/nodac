# nodac/network.py

#
# Copyright (c) 2012 Roberto D'Auria <evfirerob@gmail.com>
#
# This file is part of NODAC.
#
# NODAC is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# NODAC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NODAC. If not, see <http://www.gnu.org/licenses/>.
#

"""This module contains neural network classes."""

import random
random.seed(0)
import objgraph
from functions import FUNCTIONS

DEBUG = True

def rand(a, b):
    return (b - a) * random.random() + a

class NeuralNetwork:
    """A feed-forward neural network """

    def __init__(self):
        self._layers = []

    def generate_from_xml(self, filename):
        """Generate a neural network from a XML file."""

        pass

    def get_layers(self):
        """Returns layer list."""

        return self._layers

    def add_layer(self, size=0, function="tanh"):
        """Generate a layer of the given size and add it to the network.
        Return the generated layer.
        Args:
            size: number of neurons of the generated layer.
            function: name of the activation function.
        """

        new_layer = Layer(size, function)
        self._layers.append(new_layer)
        return new_layer

    def initialize(self):
        """Call initialize() on each layer of the network"""
        for layer in self._layers:
            layer.initialize()

    def remove_layer(self, layer_id=0):
        """Removes the layer with the given id.
        Args:
            layer_id: id of the layer to remove.
        """

        try:
            if layer_id:
                self._layers.pop(len(self._layers) - layer_id)
            else:
                self._layers.pop()
        except IndexError:
            print "Layer with id", layer_id, "doesn't exist"

    def dump(self):
        """Print the network internal state."""
        nlayer = 0
        print "Number of layers:", len(self._layers)
        for layer in self._layers:
            print "\n---- Layer", nlayer, "----"
            nlayer += 1

            print "Number of neurons:", len(layer._neurons)
            for neuron in layer._neurons:
                print "Inbound links:", neuron._in_links
                print "Outbound links:", neuron._out_links
                print "(Inbound) Weights:", neuron._weights
                print "Neuron pointer:", neuron

    def weights(self):
        layers = self._layers[:0:-1] # Reverse layers' list removing the input layer
        for layer in layers:
            for neuron in layer._neurons:
                print neuron._weights

    def run(self, inputs):
        for layer in self._layers:

            layer.activate(inputs)
            inputs = []
            for neuron in layer._neurons:
                inputs.append(neuron.get_last_activation())

        return inputs

    def backpropagate(self, outputs, N, M):
        layers = self._layers[:0:-1] # Reverse layers' list removing the input layer

        for layer in layers:
            layer.calculate_deltas(outputs)

        for layer in layers:
            layer.update_weights(N, M)

        error = 0.0
        for neuron in layers[0]._neurons:
            error += 0.5 * neuron._last_error ** 2

        return error

class Layer:
    """A layer of the neural network."""

    def __init__(self, size, function):
        self._neurons = []
        self._set_size(size)
        self._set_function(function)
        self._is_input = False
        self._is_hidden = False
        self._is_output = False

    def _set_size(self, size):
        for x in xrange(size):
            self._add_neuron()

    def _set_function(self, function):
        for neuron in self._neurons:
            neuron.set_function(function)

    def _add_neuron(self, bias=False):
        new_neuron = Neuron()
        self._neurons.append(new_neuron)
        new_neuron.set_parent(self)
        if bias:
            new_neuron._is_bias = True

    def set_input(self):
        self._is_input = True
        self._add_neuron(True)

    def set_hidden(self):
        self._is_hidden = True

    def set_output(self):
        self._is_output = True

    def connect_previous(self, previous_layer):
        """Connect the layer with the previous one creating inbound links to previous_layer."""
        for neuron in self._neurons:
            for dest in previous_layer._neurons:
                neuron.add_link('in', dest)

    def connect_next(self, next_layer):
        """Connect the layer with the next one creating outbound links to next_layer."""
        for neuron in self._neurons:
            for dest in next_layer._neurons:
                neuron.add_link('out', dest)

    def initialize(self):
        """Call init_weights() on each neuron of the layer."""
        for neuron in self._neurons:
            neuron.init_weights()

    def activate(self, inputs):
        """Activate each neuron of the layer."""
        for neuron in self._neurons:
            neuron.activate(inputs)

    def calculate_deltas(self, outputs):
        """Calculate layer's deltas."""
        if self._is_output:
            if len(outputs) != len(self._neurons):
                print "Error: dataset length doesn't match number of neurons."
                exit()

        for neuron in self._neurons:
            neuron.calculate_deltas(outputs)

    def update_weights(self, N, M):
        for neuron in self._neurons:
            neuron.update_weights(N, M)

class Neuron:
    """Represents an artificial neuron."""

    def __init__(self):
        self._function = None             # Activation function
        self._function_derivative = None  # Activation function derivative
        self._last_activation = 0         # Last activation value
        self._in_links = []               # Inbound links
        self._out_links = []              # Outbound links
        self._is_bias = False             # Bias node flag
        self._weights = []                # Inbound links' weights
        self._delta = 0                   # Neuron's deltas
        self._last_error = 0              # Neuron's last error
        self._last_change = 0             # Last weight's change
        self._parent = None               # Reference to parent layer
        self._id = None                   # Neuron's index in the layer

    def set_parent(self, parent):
        self._parent = parent
        self._id = self._parent._neurons.index(self) # Retrieves the neuron's index

    def set_function(self, function):
        """Sets the activation function.
        Args:
            function: name of the function.
        """

        self._function = FUNCTIONS[function][0]
        self._function_derivative = FUNCTIONS[function][1]

    def get_last_activation(self):
        """Returns the result of last neuron activation."""

        return self._last_activation

    def init_weights(self):
        """Randomly initializes weights of inbound links."""
        for n in xrange(len(self._in_links)):
            if self._parent._is_hidden:
                self._weights.append(rand(-0.2, 0.2))
            else:
                self._weights.append(rand(-2.0, 2.0))

    def add_link(self, direction, neuron):
        """Adds an inbound or outbound link.
        Args:
            direction: "in" for inbound links, "out" for outbound ones.
            neuron: instance of Neuron class the link points to.
        """

        if direction is "in":
            self._in_links.append(neuron)
        elif direction is "out":
            self._out_links.append(neuron)

    def activate(self, inputs):
        """Activates the neuron and stores the result."""

        # If it's a bias node, its value is 1
        if self._is_bias:
            self._last_activation = 1
        else:
            weighted_avg = 0
            for i in xrange(len(inputs)):

                # If the neuron belongs to an input layer,
                # we don't need to multiply its inputs
                if self._parent._is_input:
                    weighted_avg += inputs[i]
                else:
                    weighted_avg += inputs[i] * self._weights[i]

            # Call activation function
            self._last_activation = self._function(weighted_avg)

    def calculate_deltas(self, output):
        if self._parent._is_output:
            error = output[self._id] - self.get_last_activation()
            self._delta = self._function_derivative(self.get_last_activation()) * error
            self._last_error = error
        else:
            error = 0
            for neuron in self._out_links:
                error += neuron._delta * neuron._weights[self._id]
            self._delta = self._function_derivative(self.get_last_activation()) * error
            self._last_error = error

    def update_weights(self, N, M):
        for i in xrange(len(self._in_links)):
            change = self._delta * self._in_links[i].get_last_activation()
            self._weights[i] = self._weights[i] + N * change + M * self._last_change
            self._last_change = change

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

from nodac.functions import FUNCTIONS

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
        """Generate a layer of the given size and adds it to the network.
        Args:
            size: number of neurons of the generated layer.
            function: name of the activation function.
        """

        new_layer = Layer(size, function)
        self._layers.append(new_layer)

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


class Layer:
    """A layer of the neural network."""

    def __init__(self, size, function):
        self._neurons = []
        self._set_size(size)
        self._set_function(function)

    def _set_size(self, size):
        for x in xrange(size):
            self._add_neuron()

    def _set_function(self, function):
        for neuron in self._neurons:
            neuron.set_function(function)

    def _add_neuron(self):
        new_neuron = Neuron()
        self._neurons.append(new_neuron)

    def activate(self):
        """Activate each neuron of the layer."""

        pass


class Neuron:
    """Represents an artificial neuron """

    def __init__(self):
        self._function = None             # Activation function
        self._function_derivative = None  # Activation function derivative
        self._last_result = 0             # Last activation result
        self._in_links = []               # Inbound links
        self._out_links = []              # Outbound links
        self._weights = []                # Inbound links' weights

    def set_function(self, function):
        """Sets the activation function.
        Args:
            function: name of the function.
        """

        self._function = FUNCTIONS[function][0]
        self._function_derivative = FUNCTIONS[function][1]

    def get_last_result(self):
        """Returns the result of last neuron activation."""

        return self._last_result

    def init_weights(self):
        """Randomly initializes weights of inbound links."""
        pass

    def add_link(self, direction, neuron):
        """Adds an inbound or outbound link.
        Args:
            direction: "in" for inbound links, "out" for outbound ones.
            neuron: instance of Neuron class the link points to.
        """

        pass

    def activate(self):
        """Activates the neuron and stores the result."""

        pass

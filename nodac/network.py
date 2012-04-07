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

from nodac import functions

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


class Layer:
    """A layer of the neural network."""

    def __init__(self):
        self._neurons = []

    def generate(self, size):
        """Generate a layer of the specified size.
        Args:
            size: number of neurons of the generated layer.
        """

        for i in xrange(size):
            self._neurons.append(Neuron())

    def activate(self):
        """Activate each neuron of the layer."""

        pass


class Neuron:
    """Represents an artificial neuron """

    def __init__(self):
        self._activation_function = None  # Activation function
        self._last_result = 0             # Last activation result
        self._in_links = []               # Inbound links
        self._out_links = []              # Outbound links
        self._weights = []                # Inbound links' weights

    def set_activation_function(self, function_name):
        """Sets the activation function.
        Args:
            function_name: name of the function.
        """

        pass

    def get_last_result(self):
        """Returns the result of last neuron activation."""

        return self._last_result

    def init_weights(self):
        """Randomly initializes weights of inbound links."""
        pass

    def add_link(self, direction, neuron):
        """Adds an inbound or outbound link.
        Args:
            direction: "in" for inbound links, "out" for oubound ones.
            neuron: instance of Neuron class the link points to.
        """

        pass

    def activate(self):
        """Activates the neuron and stores the result."""

        pass

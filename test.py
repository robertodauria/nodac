# test.py

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

"""Not exactly a unittest but wait... who cares?"""
import sys

if __name__ == '__main__':
    sys.path.insert(0, '.')

from nodac import network

NN = network.NeuralNetwork()
i = NN.add_layer(2, "tanh")
i.set_input()
h = NN.add_layer(3, "tanh")
o = NN.add_layer(1, "tanh")

i.connect_next(h)
h.connect_previous(i)
h.connect_next(o)
o.connect_previous(h)

NN.initialize()
NN.dump()
NN.run([1, 2])

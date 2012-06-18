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
import cProfile

if __name__ == '__main__':
    sys.path.insert(0, '.')

from nodac import network

NN = network.NeuralNetwork()
i = NN.add_layer(2, "linear")
h = NN.add_layer(2, "tanh")
o = NN.add_layer(1, "tanh")

i.set_input()
h.set_hidden()
o.set_output()

i.connect_next(h)
h.connect_previous(i)
h.connect_next(o)
o.connect_previous(h)

NN.initialize()

patterns = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]

def train():
    for i in xrange(10000):
        error = 0
        for pat in patterns:
            inputs = pat[0]
            targets = pat[1]
            NN.run(inputs)
            error += NN.backpropagate(targets, 0.5, 0.1)
        if i % 100 == 0:
            print 'error %-.5f' % error

cProfile.run('train()')

for pat in patterns:
    print pat[0], "-->", NN.run(pat[0])


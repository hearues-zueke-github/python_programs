#! /usr/bin/python2.7

import sys

import numpy as np

from dotmap import DotMap

def change_the_values(dm):
    dm.a = 6
    dm.b[3] = 10

dm = DotMap()
dm.a = 5
dm.b = np.random.randint(0, 2, (6, ))

print("before the change")
print("dm.a: {}".format(dm.a))
print("dm.b: {}".format(dm.b))

change_the_values(dm)

print("after the change")
print("dm.a: {}".format(dm.a))
print("dm.b: {}".format(dm.b))

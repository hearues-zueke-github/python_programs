#! /usr/bin/python2.7

import numpy as np

from time import time

a = np.random.random((1000, 10000))
b = np.random.random((10000, 1000))

start_time = time()
c = np.dot(b, a)
end_time = time()

diff_time = end_time-start_time

print("diff_time: {:.3f}s".format(diff_time))

#! /usr/bin/python3.5

import numpy as np

from time import time

def sort_1(a):
    return a[np.lexsort([a[:, i] for i in range(a.shape[1]-1, -1, -1)])]

def sort_2(a):
    return a[np.argsort(np.dot(a, 256**np.arange(a.shape[1]-1, -1, -1)))]

def need_time(f, a):
    start = time()
    ret = f(*a)
    delta = time()-start
    return ret, delta

# .astype(object) works too, but is slower!
a = np.random.randint(0, 256, (1000000, 4)).astype(np.int64)

ret_1, delta_1 = need_time(sort_1, (a, ))
ret_2, delta_2 = need_time(sort_2, (a, ))

print("needed time for sort_1: {:.5f}".format(delta_1))
print("needed time for sort_2: {:.5f}".format(delta_2))

print("ret_1[:10]:\n{}".format(ret_1[:10]))
print("ret_2[:10]:\n{}".format(ret_2[:10]))

check_if_equal = np.sum(np.sum(ret_1!=ret_2, axis=1)==0)==ret_1.shape[0]
print("Are ret_1 and ret_2 equal?: {}".format("Yes!" if check_if_equal else "NO!!!"))

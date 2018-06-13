#! /usr/bin/python3.5

import numpy as np

from time import time

# Get a numpy array first
a = np.random.randint(0, 10, (5, ))

# Define a new tuple of zeros with only one number at the
# specific index
def do_add_at_idx_1(arr, idx, val):
    return arr+(((0, )*(idx))+(val, )+(0, )*(arr.shape[0]-idx-1))

# Or do it like this
def do_add_at_idx_2(arr, idx, val):
    one_val_vec = np.zeros((arr.shape[0], )).astype(arr.dtype)
    one_val_vec[idx] = val
    return arr+one_val_vec

# Or add it directly to the given array
def do_add_at_idx_3(arr, idx, val):
    arr = arr.copy()
    arr[idx] += val
    return arr

# Or use directly np.add.at function (but need a copy first)
def do_add_at_idx_4(arr, idx, val):
    arr = arr.copy()
    np.add.at(arr, idx, val)
    return arr

a_old = a.copy()
print("a: {}".format(a))
print("do_add_at_idx_1(a, 2, 5): {}".format(do_add_at_idx_1(a, 2, 5)))
print("do_add_at_idx_2(a, 2, 5): {}".format(do_add_at_idx_2(a, 2, 5)))
print("do_add_at_idx_3(a, 2, 5): {}".format(do_add_at_idx_3(a, 2, 5)))
print("do_add_at_idx_4(a, 2, 5): {}".format(do_add_at_idx_4(a, 2, 5)))
print("Was 'a' modified? {}".format("No." if np.all(a==a_old) else "YES!!"))

# Now test this different approaches
n = 100000
print("n: {}".format(n))
idxs = np.random.randint(0, a.shape[0], (n, ))
vals = np.random.randint(0, 10, (n, ))

for i in range(1, 5):
    func_name = "do_add_at_idx_{}".format(i)
    func = globals()[func_name]

    start = time()
    for idx, val in zip(idxs, vals):
        func(a, idx, val)
    delta = time()-start

    print("Taken time for func '{}': {:2.4f}s".format(func_name, delta))

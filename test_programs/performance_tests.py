#! /usr/bin/python3.6

import numpy as np

from time import time

def test_find_indices_in_other_array():
    # How many elements should be there
    n1 = 1000000
    # How many different numbers there can be
    n2 = 2000

    # Create random numbers [0, n2]
    a = np.random.randint(0, n2, (n1, ))
    # Create random numbers, for what we want to search the index
    # Should also contain only unique values as possible! (but is not a must!)
    b = np.random.permutation(np.arange(0, n2))[:np.random.randint(n2//10, n2+1)]

    # 1st approach
    start_1 = time()
    idx1 = np.where(np.any(a.reshape((-1, 1))==b, axis=1))[0]
    diff_1 = time()-start_1

    # 2nd approach
    start_2 = time()
    idx2 = np.flatnonzero(np.isin(a,b))
    diff_2 = time()-start_2

    # 3rd approach
    start_3 = time()
    b2 = np.sort(b)
    sidx = np.searchsorted(b2,a)
    sidx[sidx==len(b2)] = len(b2)-1
    idx3 = np.flatnonzero(b2[sidx]==a)
    diff_3 = time()-start_3

    print("idx1: {}".format(idx1))
    print("Taken time for approach 1: diff_1: {:1.3f}s".format(diff_1))
    
    print("idx2: {}".format(idx2))
    print("Taken time for approach 2: diff_2: {:1.3f}s".format(diff_2))

    print("idx3: {}".format(idx3))
    print("Taken time for approach 3: diff_3: {:1.3f}s".format(diff_3))


if __name__ == "__main__":
    test_find_indices_in_other_array()

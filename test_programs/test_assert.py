#! /usr/bin/python3

import numpy as np

def rotate_lst(l):
    l = list(l)
    v = l.pop(0)
    l.append(v)
    return l

assert rotate_lst([0, 1, 2])==[1, 2, 0]
assert rotate_lst([(1, 2), (3, 4), (5, 6)])==[(3, 4), (5, 6), (1, 2)]


def add(a, b):
    return a+b
    # return np.abs(a+b)
    # pass
    # return None

assert add(0, 0)==0
assert add(0, 1)==1
assert add(1, 0)==1
assert add(1, -1)==0
assert add(-1, 1)==0
assert add(-1, 0)==-1
assert add(0, -1)==-1
assert add(3, 4)==7

if __name__ == "__main__":
    print("rotate_lst([1,2,3,4,5]): {}".format(rotate_lst([1,2,3,4,5])))
    # print("Hello World!")

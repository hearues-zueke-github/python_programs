#! /usr/bin/python3.5

import sys

import numpy as np

from PIL import Image

def get_transitions(b, n):
    a = list(map(int, bin(b**n)[2:]))
    transitions = np.zeros((2, 2)).astype(np.int64)
    
    # a = list(map(int, str(b**n)))
    # transitions = np.zeros((10, 10)).astype(np.int64)

    # for k in range(1, len(a)-1):
    #     for i, j in zip(a[:-k], a[k:]):
    #         transitions[i, j] += 1
    for i, j in zip(a[:-1], a[1:]):
        transitions[i, j] += 1

    return transitions

# 1211
# 1121
# 

# b = 2
# for n in range(4, 30000, 1):
#     transitions = get_transitions(b, n)

#     print("n: {}".format(n))
#     print("transitions:\n{}".format(transitions))
#     # if np.sum(transitions<=3) == 0:
#     print("np.argmax(transitions): {}".format(np.unravel_index(np.argmax(transitions), transitions.shape)))
#         # input("Press ENTER to continue...")

t1 = get_transitions(5, 104)
t2 = get_transitions(3, 103)

t3 = t1.dot(t2)
t4 = t2.dot(t1)

print("t1:\n{}".format(t1))
print("t2:\n{}".format(t2))
print("t3:\n{}".format(t3))
print("t4:\n{}".format(t4))

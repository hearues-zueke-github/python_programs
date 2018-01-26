#! /usr/bin/python2.7

import sys

import numpy as np

def get_mix_idx(n):
    idxs = np.arange(0, n)
    idxs = np.random.permutation(idxs)
    idx_i_j = idxs[:8]

    for _ in xrange(0, n*2):
        idxs = np.random.permutation(idxs)
        idx_i_j = np.hstack((idx_i_j, idxs[:8]))

    return idx_i_j.reshape((-1, 2))

constraint_1 = lambda i1, j1, i2, j2, i3, j3: \
           i1 != j2 and j1 != i2
constraint_2 = lambda i1, j1, i2, j2, i3, j3: \
           i1 != j2 and j1 != i2 and \
           i2 != j3 and j2 != i3
constraint_3 = lambda i1, j1, i2, j2, i3, j3: \
           i1 != j2 and j1 != i2 and \
           i1 != j3 and j1 != i3
constraint_4 = lambda i1, j1, i2, j2, i3, j3: \
           i1 != j2 and j1 != i2 and \
           i2 != j3 and j2 != i3 and \
           i1 != j3 and j1 != i3
get_idx = lambda sbox, i1, j1: (lambda i2, j2: (i2, j2, sbox[i2], sbox[j2]))(sbox[i1], sbox[j1])

def mix_sbox(sbox, idx_i_j, constraint):
    sbox = sbox.copy()

    for i1, j1 in idx_i_j:
        i2, j2, i3, j3 = get_idx(sbox, i1, j1)

        if constraint(i1, j1, i2, j2, i3, j3):
            t = sbox[i1]
            sbox[i1] = sbox[j1]
            sbox[j1] = t

    return sbox

def check_sbox_same_idx(n, sbox):
    numbers = np.arange(0, n)
    equal_nums = (numbers == sbox)+0

    return np.sum(equal_nums) == 0

if __name__ == "__main__":
    n = 20

    sbox_null = np.arange(0, n)
    sbox_null += 1
    sbox_null[-1] = 0

    idx_i_j = get_mix_idx(n)
    print("idx_i_j.T:\n{}".format(idx_i_j.T))

    print("sbox_null:\n{}".format(sbox_null))
    sbox = mix_sbox(sbox_null, idx_i_j, constraint_1)
    print("sbox:\n{}".format(sbox))

    is_sbox_ok = check_sbox_same_idx(n, sbox)
    print("is_sbox_ok: {}".format(is_sbox_ok))

    constraints = [constraint_1, constraint_2, constraint_3, constraint_4]
    sboxs = [mix_sbox(sbox_null, idx_i_j, constraint) for constraint in constraints]

    for i, sbox in enumerate(sboxs):
        print("i: {}, sbox:\n{}".format(i, sbox))

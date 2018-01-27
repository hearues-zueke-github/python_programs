#! /usr/bin/python2.7

import sys

import numpy as np

def get_mix_idx(n):
    idxs = np.arange(0, n)
    idxs = np.random.permutation(idxs)
    idx_i_j = idxs[:8]

    for _ in xrange(0, n//3):
        idxs = np.random.permutation(idxs)
        idx_i_j = np.hstack((idx_i_j, idxs[:8]))

    return idx_i_j.reshape((-1, 2))

def mix_sbox(sbox, idx_i_j, max_chain):
    sbox = sbox.copy()

    for i1, j1 in idx_i_j:
        i2, j2 = i1, j1

        is_ok = True
        for k in xrange(0, max_chain):
            i2, j2 = sbox[i2], sbox[j2]
            if i1 == j2 or j1 == i2:
                is_ok = False
                break

        if is_ok:
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

    print("sbox_null:\n{}".format(sbox_null))

    sboxs = [(i, mix_sbox(sbox_null, idx_i_j, i)) for i in xrange(1, 8)]

    for i, sbox in sboxs:
        print("i: {}, sbox:\n{}".format(i, sbox))

    for k, (i_1, sbox_1) in enumerate(sboxs[:-1]):
        for i_2, sbox_2 in sboxs [k+1:]:
            print("i_1: {}, i_2: {},  {}".format(i_1, i_2, (sbox_1==sbox_2)+0))

#! /usr/bin/python2.7

import sys

import numpy as np

def get_matrix_sequence(A, B, modulo, n):
    B = B.copy()
    sequence = [B]

    m = A.shape[0]
    sequence_table = np.zeros((n, m*m)).astype(np.int)
    for i in xrange(0, n):
        B = np.dot(A, B).T % modulo
        sequence.append(B)
        sequence_table[n-1-i] = B.reshape((-1, ))

    return B, sequence_table

def find_sequence_length(sequence_table):
    n = sequence_table.shape[0]
    is_found = False
    length = 0
    for i in xrange(1, n//2):
        if np.sum(sequence_table[:i] != sequence_table[i:2*i]) == 0:
            is_found = True
            length = i
            break

    if is_found:
        return length

    return 0

modulo = 5
m = 4
A = np.random.randint(0, modulo, (m, m))
B = np.random.randint(0, modulo, (m, m))

n = 100
C, sequence_table = get_matrix_sequence(A, B, modulo, n)
length = find_sequence_length(sequence_table)

while length == 0:
    C, sequence_table_next = get_matrix_sequence(A, C, modulo, n)
    sequence_table = np.vstack((sequence_table_next, sequence_table))
    length = find_sequence_length(sequence_table)
    # print("length: {}".format(length))

print("sequence_table:\n{}".format(sequence_table))
print("length: {}".format(length))

# for i, (C, sum_0, sum_1) in enumerate(sequence):
#     print("i: {}, C:\n{}\nsum_0: {}, sum_1: {}".format(i, C, sum_0, sum_1))

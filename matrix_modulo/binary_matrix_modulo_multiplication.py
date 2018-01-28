#! /usr/bin/python2.7

import sys

import numpy as np

def get_next_matrix_sequence(A, B, modulo, n):
    B = B.copy()
    sequence = [B]

    m = A.shape[0]
    sequence_table = np.zeros((n, m*m)).astype(np.int)
    for i in xrange(0, n):
        # B = np.dot(A, B) % modulo
        # B = np.dot(B, A) % modulo
        B = np.dot(np.dot(A, B) % modulo, A) % modulo
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

def get_sequence_length(A, B, modulo):
    n = 100
    C, sequence_table = get_next_matrix_sequence(A, B, modulo, n)
    length = find_sequence_length(sequence_table)

    while length == 0:
        C, sequence_table_next = get_next_matrix_sequence(A, C, modulo, n)
        sequence_table = np.vstack((sequence_table_next, sequence_table))
        length = find_sequence_length(sequence_table)

    return length

def get_max_length(m, modulo):
    get_matrix = lambda: np.random.randint(0, modulo, (m, m))

    A_max = 0
    B_max = 0
    length_max = 0
    same_length = 0
    for i in xrange(0, 10000):
        # if i % 50 == 0:
        #     print("i: {}".format(i))
        # print("i: {}".format(i))

        A = get_matrix()
        B = get_matrix()
        length = get_sequence_length(A, B, modulo)

        if length_max <= length:
            A_max = A
            B_max = B
            if length_max == length:
                same_length += 1
            else:
                same_length = 0

            if same_length > 30:
                break
            length_max = length

    return A, B, length_max

def try_similar_matrices(m, modulo):
    A, B, length_max = get_max_length(m, modulo)

    print("A:\n{}".format(A))
    print("B:\n{}".format(B))
    print("length_max: {}".format(length_max))

    matrices_A_lengths = []
    for i in xrange(1, modulo):
        A_lengths = A.copy()
        for y in xrange(0, m):
            for x in xrange(0, m):
                A[y, x] = (A[y, x]+i) % modulo
                # print("i: {}, y: {}, x: {}".format(i, y, x))
                # print("A:\n{}".format(A))
                A_lengths[y, x] = get_sequence_length(A, B, modulo)
                A[y, x] = (A[y, x]-i) % modulo
        matrices_A_lengths.append((i, A_lengths))

    for i, A_lengths in matrices_A_lengths:
        print("i: {}, A_lengths:\n{}".format(i, A_lengths))

if __name__ == "__main__":
    m = 2
    print("m: {}".format(m))

    # try_similar_matrices(m, 7)

    # sys.exit(0)

    modulos = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29] #, 31, 37, 41, 43, 47]
    print("modulos: {}".format(modulos))

    lengths = []
    for modulo in modulos:
        _, _, length = get_max_length(m, modulo)
        print("modulo: {}, length: {}".format(modulo, length))
        lengths.append(length)

    lengths = np.array(lengths)
    print("lengths: {}".format(lengths))

    diff = lengths[1:]-lengths[:-1]
    print("diff: {}".format(diff))

    # diff = the sequence A069482 from oeis.org
    # https://oeis.org/search?q=5%2C16%2C24%2C72&sort=&language=english&go=Search
    # for: B = np.dot(A, B) % modulo

    # lengths = the sequence A084921 from oeis.org
    # https://oeis.org/search?q=3%2C4%2C12%2C24%2C60%2C84&language=english&go=Search
    # for: B = np.dot(np.dot(A, B) % modulo, A) % modulo

#! /usr/bin/python2.7

import sys

import numpy as np

from copy import deepcopy
from dotmap import DotMap

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


def get_next_matrix_sequence_func(data, func):
    """
    :param data: the specifiv values for the function func
    :type data: DotMap with specific values
    data contains:
    # :param modulo: for defining the range of the values from 0 to modulo-1
    # :type modulo: int
    # :param n: the length of the sequence
    # :type n: int
    # :param x: an array of vectors, matrices
    # :type x: list
    :param func: a specific function which returns a vector
    :type func: function with parameters
    :return: returns a matrix of the sequence values
    :rtype: narray with the shape (n, w), w is a number
    """
    n = data.n
    m = data.a.shape[0]
    sequence_table = np.zeros((n, data.length)).astype(np.int)
    for i in xrange(0, n):
        sequence_table[n-1-i] = func(data)
        # print("i: {}, data.x[1]:\n{}".format(i, data.x[1]))

    return sequence_table

def get_sequence_length_func(data, func):
    """
    :param data: the specifiv values for the function func
    :type data: DotMap with specific values
    data contains:
    # :param modulo: for defining the range of the values from 0 to modulo-1
    # :type modulo: int
    # :param n: the length of the sequence
    # :type n: int
    # :param x: an array of vectors, matrices
    # :type x: list
    :param func: a specific function which returns a vector
    :type func: function with parameters
    :return: returns only the length of a found sequence
    :rtype: int
    """
    sequence_table = get_next_matrix_sequence_func(data, func)
    length = find_sequence_length(sequence_table)

    while length == 0:
        sequence_table_next = get_next_matrix_sequence_func(data, func)
        sequence_table = np.vstack((sequence_table_next, sequence_table))
        length = find_sequence_length(sequence_table)

    return length

def get_max_length_func(data, func):
    # m = data.m
    # get_matrix = lambda: np.random.randint(0, modulo, (m, m))

    x = deepcopy(data.x)
    length_max = 0
    same_length = 0
    for i in xrange(0, 10000):
            # print("i: {}".format(i))
        # print("i: {}".format(i))

        data.x = data.new_x()
        # data.x[0] = get_matrix()
        # data.x[1] = get_matrix()
        # data.x[2] = get_matrix()
        length = get_sequence_length_func(data, func)
        
        if i % 500 == 0:
            print("i: {}, length_max: {}, same_length: {}".format(i, length_max, same_length))

        if length_max <= length:
            x = deepcopy(data.x)
            if length_max == length:
                same_length += 1
            else:
                same_length = 0

            if same_length > 10:
                break
            length_max = length

    data.x = x

    return data, length_max

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
    modulo = 7
    m = 2
    # print("m: {}".format(m))
    # print("modulo: {}".format(modulo))

    data = DotMap()
    data.m = m
    data.n = 100
    data.modulo = modulo
    # data.x = []
    # data.x.append(np.random.randint(0,  modulo, (m, m)))
    # data.x.append(np.random.randint(0,  modulo, (m, m)))
    # data.x.append(np.random.randint(0,  modulo, (m, m)))

    get_vector = lambda: np.random.randint(0, modulo, (m, ))
    get_matrix = lambda: np.random.randint(0, modulo, (m, m))
    get_tensor = lambda: np.random.randint(0, modulo, (m, m, m))
    
    def get_func_new_x_1():
        def func(data):
            data.x[1] = np.dot(data.x[0], data.x[1]) % data.modulo
            return data.x[1].reshape((-1, ))

        def new_x():
            return [get_matrix(), get_matrix()]

        return func, new_x

    def get_func_new_x_2():
        def func(data):
            data.x[1] = np.dot(np.dot(data.x[0], data.x[1]) % data.modulo, data.x[0]) % data.modulo
            return data.x[1].reshape((-1, ))

        def new_x():
            return [get_matrix(), get_matrix()]

        return func, new_x

    def get_func_new_x_3():
        def func(data):
            data.x[1] = np.dot(data.x[0], data.x[1]) % data.modulo
            data.x[2] = np.dot(data.x[1], data.x[2]) % data.modulo
            return data.x[2].reshape((-1, ))

        def new_x():
            return [get_matrix(), get_matrix(), get_matrix()]

        return func, new_x

    def get_func_new_x_4():
        def func(data):
            data.x[1] = np.dot(data.x[0], data.x[1]) % data.modulo
            # data.x[0] = np.roll(data.x[0], 1, axis=0)
            return data.x[1].reshape((-1, ))

        def new_x():
            return [get_matrix(), get_vector()]

        return func, new_x

    def get_func_new_x_5():
        def func(data):
            v0 = np.dot(data.x[0], data.x[2]) % data.modulo
            data.x[1] = np.dot(data.x[0], data.x[1].T) % data.modulo
            v1 = np.dot(data.x[1], data.x[2]) % data.modulo
            # data.x[2] = v1
            data.x[2] = (v0+v1) % data.modulo
            return data.x[2].reshape((-1, ))

        def new_x():
            return [get_matrix(), get_matrix(), get_vector()]

        return func, new_x

    def get_func_new_x_6():
        def func(data):
            new_tensor = np.dot(data.x[0], data.x[1]) % data.modulo
            data.x[1] = (data.x[1]+np.sum(new_tensor, axis=0)) % data.modulo
            data.x[1] = (data.x[1]+np.sum(new_tensor, axis=1)) % data.modulo
            data.x[1] = (data.x[1]+np.sum(new_tensor, axis=2)) % data.modulo
            return data.x[1].reshape((-1, ))

        def new_x():
            return [get_tensor(), get_matrix()]

        return func, new_x

    def get_func_new_x_7():
        def func(data):
            new_matrix_1 = np.dot(data.x[0], data.x[1]) % data.modulo
            new_matrix_2 = np.dot(data.x[1], data.x[0]) % data.modulo
            # data.x[1] = new_matrix_1**2 % data.modulo
            data.x[1] = (new_matrix_1**2+new_matrix_2) % data.modulo
            return data.x[1].reshape((-1, ))

        def new_x():
            return [get_matrix(), get_matrix()]

        return func, new_x
    
    func, new_x = get_func_new_x_7()
    data.new_x = new_x

    data.x = data.new_x()
    data.length = func(data).shape[0]

    # modulos = np.arange(2, 14).tolist() #, 17, 19, 23, 29, 31, 37, 41] #, 43, 47]
    modulos = [2, 3, 5, 7, 11, 13] #, 17, 19, 23, 29, 31, 37, 41] #, 43, 47]
    lengths = np.zeros((len(modulos), )).astype(np.int)
    for i, modulo in enumerate(modulos):
        data.modulo = modulo
        print("modulo: {}".format(modulo))

        data, max_length = get_max_length_func(data, func)
        # print("modulo: {}, max_length: {}".format(modulo, max_length))
        lengths[i] = max_length

    print("lengths: {}".format(lengths.tolist()))
    
    diff = lengths[1:]-lengths[:-1]
    print("diff: {}".format(diff.tolist()))

    sys.exit(0)

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

    # lengths = the sequence A127917 from oeis.org
    # https://oeis.org/search?q=6%2C+24%2C+120%2C+336&sort=&language=english&go=Search
    # for: get_func_new_x_6

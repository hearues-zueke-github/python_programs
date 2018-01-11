#! /usr/bin/python2.7

import sys

import numpy as np

def get_all_numbers(m, n):
    X = np.zeros((n**m, m)).astype(np.uint)

    X[:n, -1] = np.arange(0, n)
    for i in xrange(2, m+1):
        for j in xrange(1, n):
            X[n**(i-1)*j: n**(i-1)*(j+1), -i] = j
            X[n**(i-1)*j: n**(i-1)*(j+1), -i+1:] = X[: n**(i-1), -i+1:]

    return X

def test_X(X, m, n):
    powers = np.arange(m-1, -1, -1)
    sums_increased = np.sum((X*n**powers).astype(np.uint), axis=1)
    print("sums_increased: {}".format(sums_increased))

    diffs = sums_increased[1:]-sums_increased[:-1]
    print("diffs: {}".format(diffs))
    diffs_sum = np.sum(diffs-1)
    print("diffs_sum: {}".format(diffs_sum))

def find_new_A(X, m, n):
    A = np.random.randint(0, n, (m, m))
    b = np.random.randint(0, n, (m, ))
    b[:] = 0

    Y = (np.dot(X, A)+b).astype(np.uint)%n

    y_numbers = np.sum((Y*n**np.arange(m-1, -1, -1)), axis=1).astype(np.uint)
    # print("y_numbers: {}".format(y_numbers))

    y_numbers_sorted = np.sort(y_numbers)
    # print("y_numbers_sorted: {}".format(y_numbers_sorted))

    y_diffs = y_numbers_sorted[1:]-y_numbers_sorted[:-1]
    # print("y_diffs: {}".format(y_diffs))

    all_ones = np.sum(y_diffs!=1)==0
    if all_ones:
        # dotmap = DotMap()
        return A,
        # return A, b
    # tuples = ((tuple(x), tuple(y)) for x, y in zip(X, Y))

    return None

def does_list_contains_duplicate(As, A):
    for i in xrange(0, m):
        Av = np.vstack((A[i:], A[:i]))
        if Av.tolist() in As:
            return True
            
        for i in xrange(1, m):
            Ah = np.hstack((Av[:, i:].reshape((-1, m-i)), Av[:, :i].reshape((-1, i))))
            if Ah.tolist() in As:
                return True
    
    return False

def test_A_properties(X, A, m, n):
    print("A:\n{}".format(A))
    b = np.random.randint(0, n, (m, ))
    print("b:\n{}".format(b))

    def get_y_diffs(A):
        Y = (np.dot(X, A)+b).astype(np.uint)%n
        y_numbers = np.sum((Y*n**np.arange(m-1, -1, -1)), axis=1).astype(np.uint)
        y_numbers_sorted = np.sort(y_numbers)
        y_diffs = y_numbers_sorted[1:]-y_numbers_sorted[:-1]
        return y_numbers, y_diffs

    As_y_nums_diffs = []

    for i in xrange(0, m):
        Av = np.vstack((A[i:], A[:i]))
        print("Av{}:\n{}".format(i, Av))

        y_numbers, y_diffs = get_y_diffs(Av)
        As_y_nums_diffs.append((Av, y_numbers, y_diffs))

        for i in xrange(1, m):
            Ah = np.hstack((Av[:, i:].reshape((-1, m-i)), Av[:, :i].reshape((-1, i))))
            print("Ah{}:\n{}".format(i, Ah))
            y_numbers, y_diffs = get_y_diffs(Ah)
            As_y_nums_diffs.append((Ah, y_numbers, y_diffs))

    for A, y_num, y_diff in As_y_nums_diffs:
        print("A:\n{}".format(A))
        print("b: {}".format(b))
        print("y_num: {}".format(y_num))
        print("y_diff: {}".format(y_diff))

    sys.exit(0)

if __name__ == "__main__":
    m = 3
    n = 4

    X = get_all_numbers(m, n)

    # print("X:\n{}".format(X))

    As = []
    As_list = []
    As_list_no_dup = []

    for i in xrange(1, 60000+1):
        elements = find_new_A(X, m, n)

        if elements != None:
            # As.append(elements[0])
            A = elements[0]
            test_A_properties(X, A, m, n)

            A_list = A.tolist()
            if not A_list in As_list:
                As_list.append(A_list)
                As.append(A)

            # if not dist_no_dup.append(A.tolist())

        if i % 1000 == 0:
            print("i: {}".format(i))
            print("found all matrices: {}".format(len(As_list)))
            print("found unique matrices: {}".format(len(As_list_no_dup)))

    # for i, A in enumerate(As):
    #     print("A{}:\n{}".format(i, A))

    print("m: {}".format(m))
    print("n: {}".format(n))
    print("found all matrices: {}".format(len(As_list)))
    print("found unique matrices: {}".format(len(As_list_no_dup)))

    # for t in tuples:
    #     print("t: {}".format(t))

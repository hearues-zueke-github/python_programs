#! /usr/bin/python2.7

import numpy as np

import matplotlib.pyplot as plt

max_num = 3

# x = 8
# y = 2
def find_max_plus_one_similars(x, y):
    masks = np.zeros((x*y, y+2, x+2))
    cross = np.zeros((3, 3))
    cross[(0, 1, 1, 2), (1, 0, 2, 1)] = 1
    cross[1, 1] = -4

    # print("x = {}".format(x))
    # print("y = {}".format(y))
    # print("cross = {}".format(cross))

    for j in xrange(0, y):
        for i in xrange(0, x):
    #         print("j: {}, i: {}".format(j, i))
            masks[j*x+i, j:j+3, i:i+3] = cross
    #         print("masks =\n{}".format(masks[j*x+i]))

    plus = 30
    n1 = np.zeros((y, x)) # np.random.randint(0, max_num+plus+1, (y, x))
    n2 = np.ones((y, x)) # np.random.randint(0, max_num+plus+1, (y, x))
    # print("n1:\n{}".format(n1))
    # print("n2:\n{}".format(n2))

    def applied_sandpiles(max_num, masks, n):
        bigger = np.where(n[1:-1, 1:-1] > max_num)

        while len(bigger[0]) > 0:
            n += np.sum(masks[bigger[0]*x+bigger[1]], axis=0)
            bigger = np.where(n[1:-1, 1:-1] > max_num)

        return n[1:-1, 1:-1].astype(np.int)

    def add_two_sandpiles(max_num, n1, n2):
        rows, cols = n1.shape
        temp1 = np.zeros((rows+2, cols+2))
        temp2 = np.zeros((rows+2, cols+2))

        temp1[1:-1, 1:-1] = n1
        temp2[1:-1, 1:-1] = n2

        calc = temp1 + temp2
        bigger = np.where(calc[1:-1, 1:-1] > max_num)

        while len(bigger[0]) > 0:
            contra = np.sum(masks[bigger[0]*x+bigger[1]], axis=0)
            calc += contra
            bigger = np.where(calc[1:-1, 1:-1] > max_num)

        sandpile = applied_sandpiles(max_num, masks, calc)
        # print("finish calc:\n{}".format(sandpile))
        return sandpile

    iterations = 800
    added = np.zeros((iterations, y, x)).astype(np.int)
    added[0] = add_two_sandpiles(max_num, n1, n2+i)
    for i in xrange(1, iterations):
        added[i] = add_two_sandpiles(max_num, added[i-1], np.ones((y, x)))
        # print("i: {}, added[i]:\n{}".format(i, added[i]))

    # Find first equal!
    founded = []
    for i, b in enumerate(added):
        if np.sum(b==0) == x*y:
            # print("continue!")
            continue
        amount = 1
        b = b.copy()
        added[i] = 0
        for j, a in enumerate(added):
            # print("is equal to a[0]?\n{}".format(np.hstack((added[0], a))))
            # raw_input()
            if np.sum(a == b) == x*y:
                added[j] = 0
                amount += 1
                # print("Found same!")
        founded.append((b, amount))

    # print("founded:\n{}".format(founded))
    print("x: {}, y: {}, len(founded): {}".format(x, y, len(founded)))
    # for a, c in founded:
    #     print("{}".format(a))

    return len(founded)

lens = map(lambda y: map(lambda x: (x, y, find_max_plus_one_similars(x, y)), xrange(y, 5)), range(1, 5))

lens = reduce(lambda a, b: a+b, lens, [])

print("lens = {}".format(lens))

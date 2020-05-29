#! /usr/bin/python3

import sys

import numpy as np

import matplotlib.pyplot as plt

from functools import reduce

MAX_NUM = 1

def applied_sandpiles(max_num, masks, n):
    bigger = np.where(n[1:-1, 1:-1]>max_num)

    x = masks.shape[2]-2
    while len(bigger[0]) > 0:
        n += np.sum(masks[bigger[0]*x+bigger[1]], axis=0)
        bigger = np.where(n[1:-1, 1:-1]>max_num)

    return n[1:-1, 1:-1].astype(np.int)


def add_two_sandpiles(max_num, masks, n1, n2):
    rows, cols = n1.shape
    temp1 = np.zeros((rows+2, cols+2))
    temp2 = np.zeros((rows+2, cols+2))

    temp1[1:-1, 1:-1] = n1
    temp2[1:-1, 1:-1] = n2

    calc = temp1 + temp2
    bigger = np.where(calc[1:-1, 1:-1]>max_num)

    x = masks.shape[2]-2
    while len(bigger[0]) > 0:
        contra = np.sum(masks[bigger[0]*x+bigger[1]], axis=0)
        calc += contra
        bigger = np.where(calc[1:-1, 1:-1]>max_num)

    sandpile = applied_sandpiles(max_num, masks, calc)
    # print("finish calc:\n{}".format(sandpile))
    return sandpile


def find_max_plus_one_similars(x, y):
    print("x: {}, y: {}".format(x, y))
    masks = np.zeros((x*y, y+2, x+2))
    cross = np.zeros((3, 3))
    cross[(0, 1, 1, 2), (1, 0, 2, 1)] = 1
    cross[1, 1] = -4

    for j in range(0, y):
        for i in range(0, x):
            masks[j*x+i, j:j+3, i:i+3] = cross
    print("masks:\n{}".format(masks))

    plus = 30
    n1 = np.zeros((y, x)).astype(np.int) # np.random.randint(0, max_num+plus+1, (y, x))
    n2 = np.ones((y, x)).astype(np.int) # np.random.randint(0, max_num+plus+1, (y, x))
    # n2[:] = 0
    # n2[0, 0] = 1
    # print("n2:\n{}".format(n2))

    # print("n1:\n{}".format(n1))
    # print("n2:\n{}".format(n2))

    iterations = 30000
    added = np.zeros((iterations, y, x)).astype(np.int)
    added[0] = add_two_sandpiles(MAX_NUM, masks, n1, n2)
    # added[0] = add_two_sandpiles(MAX_NUM, masks, n1, n2+i)
    for i in range(1, iterations):
        added[i] = add_two_sandpiles(MAX_NUM, masks, added[i-1], np.ones((y, x)))
        # print("i: {}, added[i]:\n{}".format(i, added[i]))

    # if x==3 and y==3:
    #     globals()['added'] = added
    #     sys.exit(0)

    # # Find first equal!
    # l_founded = []
    # founded = []
    # for i, b in enumerate(added, 0):
    #     if np.all(b==0):
    #     # if np.sum(b==0)==x*y:
    #         # print("continue!")
    #         continue
    #     amount = 1
    #     b = b.copy()
    #     added[i] = 0
    #     l_diffs = []
    #     for j, a in enumerate(added):
    #         # print("is equal to a[0]?\n{}".format(np.hstack((added[0], a))))
    #         # raw_input()
    #         if np.sum(a==b)==x*y:
    #             added[j] = 0
    #             amount += 1
    #             # l_diffs.append(j-i)
    #             # print("Found same!")
    #     founded.append((b, amount))
    #     # founded.append((b, amount, l_diffs))
    l_founded = []
    is_found = False
    for i, b in enumerate(added, 0):
        if np.all(b==0):
            continue
        for j, a in enumerate(added[i+1:], i+1):
            if np.all(a==b):
            # if np.sum(a==b)==x*y:
                is_found = True
                break
        if is_found:
            l_founded.extend([c for c in added[i:i+j]])
            break

    # print("founded:\n{}".format(founded))
    # print("x: {}, y: {}, len(l_founded): {}".format(x, y, len(l_founded)))
    # print("l_founded: {}".format(l_founded))
    
    # print("x: {}, y: {}, len(founded): {}".format(x, y, len(founded)))
    # print("founded: {}".format(founded))
    # for a, c in founded:
    #     print("{}".format(a))

    # return len(founded)
    return len(l_founded)


if __name__=='__main__':
    print("MAX_NUM: {}".format(MAX_NUM))
    max_n = 3
    l_lens = list(map(lambda y: list(map(lambda x: (x, y, find_max_plus_one_similars(x, y)), range(y, max_n))), range(1, max_n)))

    lens = list(reduce(lambda a, b: a+b, l_lens, []))

    print("lens = {}".format(lens))

    d = {(x, y): v for x, y, v in lens}
    l = [d[(j, i)] for j in range(1, max_n) for i in range(1, j+1)]
    print("l: {}".format(l))

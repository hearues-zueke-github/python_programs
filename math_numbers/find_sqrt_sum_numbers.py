#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import numpy as np

from sympy.ntheory import factorint

def is_perfect_square(n):
    if n <= 1:
        return False

    x = n//2
    xs = [x]
    while x * x != n:
        x = (x+(n//x))//2
        if x in xs:
            return False
        xs.append(x)

    return True

if __name__ == "__main__":
    jumps_idx_lst_own = [
        [1, [24]],
        [3, [3249]],
        [8, [64]],
        [13, [49]],
        [14, [4]],
        [30, [484]],
        [194, [361]],
        [367, [50]],
        [432, [4]],
        [1349, [289]],
        [1554, [81]],
        [1932, [9]],
        [5082, [49]],
        [6791, [568]],
        [11928, [441]],
        [12958, [4]],
        [15254, [64]],
    ]

    jumps_idx_lst = []
    jumps = []
    # jumps = [1349, 1554, 1932]#+list(range(1933, 10001))
    # jumps += [1, 3, 8, 13, 14, 30, 194, 367, 432, 1349,
    #     1554, 1932, 5082, 6791, 11928, 12958, 15254]
    jumps += list(range(15254+1, 30001))

    for start_num in range(1, 2):
        for jump in jumps:
            # print("\njump: {}".format(jump))
            if jump % 100 == 0:
                print("jump: {}".format(jump))
            ns = np.arange(0, 3000)*jump+1
            # ns = np.arange(1, 1+jump*100000, jump)
            # ns = np.arange(1, 1000001, jump)
            sums = np.cumsum(ns**2)
            # def is_prime(n):
            #     if len(factorint(n)) == 1:
            #         return True
            #     return False

            idx_square_nums = np.vectorize(is_perfect_square)(sums)
            # print("idx_square_nums:\n{}".format(idx_square_nums))

            amount_squares = np.sum(idx_square_nums)
            if amount_squares > 0:
                print("\nstart_num: {}".format(start_num))
                print("jump: {}".format(jump))

                squares_idx = np.where(idx_square_nums)[0]+1
                squares_sums = sums[idx_square_nums]

                jumps_idx_lst.append([jump, squares_idx.tolist()])

                print("squares_idx: {}, squares_sums: {}".format(squares_idx, squares_sums))
                print("Nums to be squared and take the sum:")
                for idx in squares_idx:
                    part_ns = ns[:idx]
                    print("idx: {}, ns[:idx]: {}{}".format(idx, part_ns[:10], "..." if part_ns.shape[0] > 10 else ""))
                print("amount_squares: {}".format(amount_squares))
            # else:
            #     print("No perfect sqrt root sums found!")
    # print("jumps_idx_lst:\n{}".format(jumps_idx_lst))
    print("jumps_idx_lst_own = [")
    for lst in jumps_idx_lst:
        print("    {},".format(lst))
    print("]")

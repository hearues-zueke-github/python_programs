#! /usr/bin/python3.6

import sys

import matplotlib.pyplot as plt

import numpy as np

# TODO: make a own class only for e.g. 2,3,5,7,11,13 prime calculator (multiprocesses!)

def get_first_prims(primes):
    prod = np.multiply.reduce(primes)
    prim_nums = np.arange(1, prod)
    first_prims = np.where(np.all(prim_nums.reshape((-1, 1))%primes!=0, axis=1))[0]+1
    return first_prims, prod

def get_primes_from_first_prims(first_primes, first_prims, prod, n_parts=10, n_max=None):
    primes = first_primes.copy()
    not_primes = np.array([], dtype=np.int64)

    # TODO: better first prime calculation!
    # if first_primes.shape[0] >= 6:
        # print("prod: {}".format(prod))
        # print("first_prims: {}".format(first_prims))
        # print("first_primes: {}".format(first_primes))

    max_p_sqrt = primes[-1]**2

    if n_parts == 1:
        first_prims_cpy = first_prims[1:].copy()
    else:
        first_prims_cpy = ((np.arange(0, n_parts)*prod).reshape((-1, 1))+first_prims).reshape((-1, ))[1:]

    times = 1
    while first_prims_cpy.shape[0] > 0:
    # while last_num > max_p_sqrt:
        print("\ntimes: {}".format(times))
        print("first_prims_cpy.shape: {}".format(first_prims_cpy.shape))
        print("primes.shape: {}".format(primes.shape))
        print("primes[-1]: {}".format(primes[-1]))
        print("primes[-1]**2: {}".format(primes[-1]**2))
        print("max_p_sqrt: {}".format(max_p_sqrt))

        amount = np.sum(primes**2<=first_prims_cpy[-1])
        print("amount: {}".format(amount))
        idx_first_iter_prim = np.all(first_prims_cpy.reshape((-1, 1))%primes[:amount]!=0, axis=1)
        first_prims_cpy_2 = first_prims_cpy[idx_first_iter_prim]
        first_iter_none_prim = first_prims_cpy[~idx_first_iter_prim]
        not_primes = np.hstack((not_primes, first_iter_none_prim))
        # print("first_prims_cpy: {}".format(first_prims_cpy))
        # print("first_iter_none_prim: {}".format(first_iter_none_prim))

        first_prims_cpy = first_prims_cpy
        first_prims_true_prim = first_prims_cpy[first_prims_cpy < max_p_sqrt]
        primes = np.hstack((primes, first_prims_true_prim))
        # print("first_prims_true_prim: {}".format(first_prims_true_prim))
        # print("primes: {}".format(primes))
        # print("times: {}".format(times))
        times += 1
        idx_2 = np.isin(first_prims_cpy, first_prims_true_prim)
        first_prims_cpy = first_prims_cpy[~idx_2]
        max_p_sqrt = primes[-1]**2

            # print("first_prims_cpy: {}".format(first_prims_cpy))
            # print("max_p_sqrt: {}".format(max_p_sqrt))
            
            # if times > 2:
            #     sys.exit(-123)

        # sys.exit(-1900)

    # primes = primes.tolist()
    # for p in first_prims[1:]:
    #     if np.all(p%primes!=0):
    #         primes.append(p)
    #     else:
    #         not_primes.append(p)

    # primes = np.array(primes)
    # not_primes = np.array(not_primes, dtype=np.int64)

    i_choosen = 0
    n_choosen = primes[i_choosen]

    if n_parts == 1:
        next_prims = np.array([4])
    else:
        next_prims = ((np.arange(1, n_parts)*prod).reshape((-1, 1))+first_prims).reshape((-1, ))
        # for k in range(1, n_parts):
        #     next_prims = k*prod+first_prims
        #     last_num = next_prims[-1]

        #     while n_choosen**2 < last_num:
        #         i_choosen += 1
        #         n_choosen = primes[i_choosen]

        #     real_primes_idx = np.all(next_prims.reshape((-1, 1))%primes[:i_choosen]!=0, axis=1)
        #     real_primes = next_prims[real_primes_idx]
        #     not_real_primes = next_prims[~real_primes_idx]

        #     primes = np.hstack((primes, real_primes))
        #     not_primes = np.hstack((not_primes, not_real_primes))
    
    j = 1

    length_primes = [primes.shape[0]]
    length_not_primes = [not_primes.shape[0]]

    while next_prims[0] < n_max:
        last_num = next_prims[-1]

        print("j: {}".format(j))
        while n_choosen**2 < last_num:
            i_choosen += 1
            n_choosen = primes[i_choosen]
        
        real_primes_idx = np.all(next_prims.reshape((-1, 1))%primes[:i_choosen]!=0, axis=1)
        real_primes = next_prims[real_primes_idx]
        not_real_primes = next_prims[~real_primes_idx]

        primes = np.hstack((primes, real_primes))
        not_primes = np.hstack((not_primes, not_real_primes))

        length_primes.append(primes.shape[0])
        length_not_primes.append(not_primes.shape[0])

        # next_prims = j*prod+first_prims
        next_prims = ((np.arange(n_parts*j, n_parts*(j+1))*prod).reshape((-1, 1))+first_prims).reshape((-1, ))
        j += 1

    length_primes = np.array(length_primes)
    length_not_primes = np.array(length_not_primes)

    l_p = primes.shape[0]
    l_np = not_primes.shape[0]
    print("l_np/(l_p+l_np): {:.3f}%".format(l_np/(l_p+l_np)*100))

    return length_primes, length_not_primes


if __name__ == "__main__":
    sys.exit()

    # first_primes = np.array([2, 3, 5, 7, 11, 13, 17, 19])
    # prods = 2*3*5*7*11*13*17*19
    # prods = 2*3*5*7*11*13*17
    # prods = 2*3*5*7*11*13
    
    # first_primes = np.array([2, 3, 5, 7, 11, 13, 17, 19])
    # first_primes = np.array([2, 3, 5, 7])
    # prods = 2*3*5*7

    ls = []
    for i  in range(2, 7+1):
        print("i: {}".format(i))
        first_primes_part = first_primes[:i]
        first_prims, prod = get_first_prims(first_primes_part)
        ls_p, ls_np = get_primes_from_first_prims(first_primes_part, first_prims, prod, n_parts=prods//prod, n_max=prods*20)
        ls.append((ls_p, ls_np))

    print("len(ls): {}".format(len(ls)))
    sys.exit()

    plt.figure()

    plt.title("Graph: primes/(primes+not_primes)")
    
    ps = []
    ps_legend = []
    colors = ["#00AA00",
              "#AA0000",
              "#00AAFF",
              "#FFAA00",
              "#FF00AA",
              "#FFAAAA",
              "#AAAAAA",
              "#FF00AA",
              "#FF00AA",
              "#FF00AA"]
    for i in range(0, 6):
        ls_p, ls_np = ls[i]
        xs = np.arange(1, ls_p.shape[0]+1).astype(np.float)/ls_p.shape[0]
        ys = ls_p/(ls_p+ls_np)*100

        xs = np.hstack(((0, ), xs))
        ys = np.hstack(((100, ), ys))

        p = plt.plot(xs, ys, color=colors[i])[0]
        ps.append(p)
        ps_legend.append(", ".join(list(map(str, first_primes[:i+2].tolist()))))
    
    plt.legend(ps, ps_legend)
    # plt.legend([p1], ["2, 3"])

    plt.ylim([-1., 101.])

    plt.show()

    # n = 10
    # prims = np.arange(0, n).reshape((-1, 1))*prod+first_prims

    # print("prims:\n{}".format(prims))

#! /usr/bin/python2.7

import pylab
import sys

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def lcd(a, b):
    if b > a:
        a, b = b, a

    while b > 0:
        t = a % b
        a = b
        b = t

    return a

def get_real_prime_numbers(max_n):
    primes = [2, 3, 5]

    x = 7
    idx = 0
    incs = [2, 4]
    while x <= max_n:
        is_prime = True
        y = int(np.sqrt(x))+1

        for i in primes:
            if i > y:
                break

            if x % i == 0:
                is_prime = False
                break

        if is_prime:
            primes.append(x)

        idx = (idx+1) % 2
        x += incs[idx]

    return primes

def get_pseudo_prime_numbers(n, k, d):
    assert n >= 2
    assert k >= 1
    assert d >= 1

    pseudo_primes = []

    if d != 1:
        pseudo_primes.append(d)

    if len(pseudo_primes) == 0 or \
       (k+d) % pseudo_primes[0] != 0:
        pseudo_primes.append(k+d)

    x = len(pseudo_primes)
    while len(pseudo_primes) < n:
        y = k*x+d
        x += 1

        # max_num = int(np.sqrt(y))+1
        is_prime = True
        for i in pseudo_primes:
            # if i > max_num:
            #     break

            if y % i == 0:
                is_prime = False
                break

        if is_prime:
            pseudo_primes.append(y)

    return pseudo_primes

def get_split_prime_pseudo_prime(pseudo_primes, real_primes):
    split_primes = [[], []]

    for x in pseudo_primes:
        if x in real_primes:
            split_primes[0].append(x)
        else:
            split_primes[1].append(x)

    return split_primes

def show_plot_percentage_real_primes(data, n):
    data_arr = np.array(data)
    k_max = np.max(data_arr[:, 0])
    d_max = np.max(data_arr[:, 1])

    print("k_max: {}".format(k_max))
    print("d_max: {}".format(d_max))

    matrix = np.zeros((k_max, d_max))
    for k, d, len_rp in data:
        matrix[k-1, d-1] = len_rp

    matrix /= n

    x = -0.4
    y = 0.2

    fig, ax = plt.subplots(figsize=(12, 9))
    res = ax.imshow(pylab.array(matrix), cmap=plt.cm.jet, interpolation='nearest')
    for i, line in enumerate(matrix):
        for j, val in enumerate(line):
            # if val > 0:
            # plt.text(j+x, i+y, "{:.02f}".format(val), fontsize=7)
            pass

    plt.title("Percentage of real primes, n: {}".format(n), y=1.08)
    plt.xlabel("d")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.ylabel("k")

    ax.xaxis.set_major_locator(ticker.FixedLocator((xrange(0, d_max))))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter((xrange(1, d_max+1))))

    ax.yaxis.set_major_locator(ticker.FixedLocator((xrange(0, k_max))))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter((xrange(1, k_max+1))))

    plt.subplots_adjust(bottom=0.1, top=0.85)#left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    cb = fig.colorbar(res)
    # plt.savefig(network_path+"/"+plots_file_name_prefix+"_confusion_matrix_{}.png".format(suffix), format="png")
    # plt.close()
    plt.show()

if __name__ == "__main__":
    pseudo_primes_lst = []
    n = 1000
    max_k = 30
    max_d = 30
    for k in xrange(1, max_k+1):
        for d in xrange(1, max_d+1):
            if k >= 1 and d >= 1 and lcd(k, d) == 1:
                pseudo_primes = get_pseudo_prime_numbers(n, k, d)
                pseudo_primes_lst.append((k, d, pseudo_primes))
                # print("k: {}, d: {}:\n{}".format(k, d, pseudo_primes))
                # print("len(pseudo_primes): {}".format(len(pseudo_primes)))
                print("got: k: {}, d: {}".format(k, d))

    pseudo_primes_l = list(map(lambda x: x[2], pseudo_primes_lst))
    # print("pseudo_primes_l:\n{}".format(pseudo_primes_l))
    arr_pseudo_primes = np.array(pseudo_primes_l)
    # print("arr_pseudo_primes.shape: {}".format(arr_pseudo_primes.shape))
    max_num = np.max(arr_pseudo_primes)
    # print("max_num: {}".format(max_num))

    real_primes = get_real_prime_numbers(max_num)
    # print("real_primes:\n{}".format(real_primes))

    # split_primes_lst = list(map(lambda l: get_split_prime_pseudo_prime(l, real_primes), pseudo_primes_l))

    data = []
    for k, d, ps in pseudo_primes_lst:
    # for (k, d, ps), (rp, pp) in zip(pseudo_primes_lst, split_primes_lst):
        rp, pp = get_split_prime_pseudo_prime(ps, real_primes)
        data.append((k, d, len(rp)))
        print("k: {}, d: {}, len(rp): {}, len(pp): {}".format(k, d, len(rp), len(pp)))

    show_plot_percentage_real_primes(data, n)

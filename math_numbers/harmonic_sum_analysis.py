#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import sys

import numpy as np

from math import gcd
from math import factorial as fac

import decimal
prec = 100
decimal.getcontext().prec = prec
from decimal import Decimal as Dec
Decimal = Dec

import matplotlib.pyplot as plt

path_dir_root = os.path.dirname(os.path.abspath(__file__))+"/"

# TODO: create a function, which is much much faster for approximating
# the log function!

# abs(x) < 1
def log_dec(x, ns=1000):
    # using log(1-x) function!
    if not isinstance(x, Dec):
        x = Dec(x)
    assert x >= Dec("0") and x <= Dec("1")
    if x >= Dec("1"):
        x -= Dec("0.0000000000000001")
    x = Dec(1)-x
    s = Dec(0)
    for i in range(1, ns+1):
        s -= x**i/Dec(i)
    return s


# n...max iterations
def calc_e(n):
    s = Dec(0)
    for i in range(0, n+1):
        s += 1/Dec(fac(i))
    return s
e = calc_e(100)


def log_dec_num(x):
    if not isinstance(x, Dec):
        x = Dec(x)
    assert x > Dec("0")

    y_start = Dec(np.log(float(x)))

    print("y_start: {}".format(y_start))
    diff = x-e**y_start
    print("diff: {}".format(diff))


def get_lst_from_dict(d):
    keys = sorted(list(d.keys()))
    return [d[key] for key in keys]

if __name__ == "__main__":
    # ps = [(i, Dec(1)+1/Dec(2)**i) for i in range(0, 5)]
    ps = [(i, Dec(1)+1/Dec(2)**i) for i in range(0, 5)]

    folder_objs = "objs/"
    if not os.path.exists(folder_objs):
        os.makedirs(folder_objs)
    objs_file_path = folder_objs+"harmonic_objs.pkl.gz"

    if not os.path.exists(objs_file_path):
        objs = {}
        with gzip.open(objs_file_path, "wb") as fout:
            dill.dump(objs, fout)
    else:
        with gzip.open(objs_file_path, "rb") as fin:
            objs = dill.load(fin)

    def calc_harmonic_sequence(idx_begin, idx_end, p=Dec(1)):
        print("idx_begin: {}, idx_end: {}".format(idx_begin, idx_end))
        return [1/i**p for i in range(idx_begin, idx_end)]
        # return [1/Dec(i)**p for i in range(idx_begin, idx_end)]
    def calc_2_n_partition_harmonic_sequences(idx, n_begin=0, n_end=11, p=Dec(1)):
        print("n_begin: {}, n_end: {}, p: {}".format(n_begin, n_end, p))
        lst = []
        # p = Dec("2")
        for i in range(n_begin, n_end):
            idx_begin = 2**i
            idx_end = 2**(i+1)
            # print("idx_begin: {}, idx_end: {}".format(idx_begin, idx_end))
            t = (idx, prec)
            if not i in objs[t]:
                sum_lst_part = np.sum(calc_harmonic_sequence(idx_begin, idx_end, p=p))
                objs[t][i] = sum_lst_part
            lst.append(objs[t][i])

        return np.array(lst)

    # lst = []
    # for idx, p in ps:
    #     t = (idx, prec)
    #     if not t in objs:
    #         objs[t] = {}
    #     lst_part = calc_2_n_partition_harmonic_sequences(idx, n_begin=0, n_end=22, p=p)
    #     lst.append((idx, p, lst_part))
    # # print("lst:\n{}".format(lst))

    # sums = [(i, np.sum(l)) for i, p, l in lst]
    # print("sums:\n{}".format(sums))

    # with gzip.open(objs_file_path, "wb") as fout:
    #     dill.dump(objs, fout)

    # qs = ys[1:]/ys[:-1]
    # ds = qs[1:]-qs[:-1]

    # lsts = []
    # i_max = 14
    # base = 2
    # for i in range(-1, 1):
    #     p = Dec("1")+1/Dec(2**i)
    #     if i == -1:
    #         p = Dec("1")
    #     lst = calc_harmonic_sequence(1, base**i_max, p=p)
    #     arr = np.array(lst)
    #     print("p: {}".format(p))
    #     lst_sum_begin = []
    #     lst_sum = []
    #     for j in range(0, i_max):
    #         arr_part_begin = arr[:base**(j+1)-1]
    #         arr_part = arr[base**j-1:base**(j+1)-1]
    #         print("j: {}, arr_part.shape: {}".format(j, arr_part.shape))
    #         s_begin = np.sum(arr_part_begin)
    #         s = np.sum(arr_part)
    #         lst_sum_begin.append(s_begin)
    #         lst_sum.append(s)
    #     lsts.append((i, p, lst_sum_begin, lst_sum))

    p = Dec(1)+1/Dec(2**8)

    print("p: {}".format(p))
    lst_sum_begin = []
    lst_sum_begin_max = []
    lst_sum_begin_min = []
    lst_sum = []
    lst_sum_max = []
    lst_sum_min = []
    s_begin_min = Dec(0)
    s_begin_max = Dec(0)
    lst = calc_harmonic_sequence(1, 2**16, p=p)
    for n in range(0, 15):
        arr = np.array(lst)
        print("p: {}".format(p))
        # lst_sum_begin = []
        # lst_sum = []

        arr_part_begin = arr[:2**(n+1)-1]
        arr_part = arr[2**n-1:2**(n+1)-1]
        print("n: {}, arr_part.shape: {}".format(n, arr_part.shape))
        s_begin = np.sum(arr_part_begin)
        s = np.sum(arr_part)
        lst_sum_begin.append(s_begin)
        lst_sum.append(s)

        s_max = 1/Dec(2**n)**(p-1)
        s_min = 2**n/Dec(2**(n+1)-1)**p

        s_begin_max += s_max
        s_begin_min += s_min

        lst_sum_begin_min.append(s_begin_min)
        lst_sum_begin_max.append(s_begin_max)
        lst_sum_min.append(s_min)
        lst_sum_max.append(s_max)

    print("p: {}".format(p))
    print("lst_sum_begin_min[-1]: {}".format(lst_sum_begin_min[-1]))
    print("lst_sum_begin_max[-1]: {}".format(lst_sum_begin_max[-1]))

    arr_min = np.array(lst_sum_begin_min)
    arr = np.array(lst_sum_begin)
    arr_max = np.array(lst_sum_begin_max)

    qs = (arr_max-arr)[1:]/(arr-arr_min)[1:]
    print("qs: {}".format(qs))

    # print("qs: {}".format(qs))
    # print("ds: {}".format(ds))

    def get_random_color(fix_r=None, fix_g=None, fix_b=None):
        i_r, i_g, i_b = np.random.randint(64, 192, (3, ))
        # i_r, i_g, i_b = np.random.randint(0, 256, (3, ))
        return "#{:02X}{:02X}{:02X}".format(i_r, i_g, i_b)

    colors = [get_random_color() for _ in range(0, 3)]

    xs = np.arange(0, len(lst_sum_min))

    plts = []
    legend = ["sum min for p", "p = {}".format(p), "sum max for p"]
    plt.figure()

    plt.title("Sum beginnning, p = {}, watch min max values!".format(p))

    plts.append(plt.plot(xs, lst_sum_begin_min, "-", marker=".", color=colors[0])[0])
    plts.append(plt.plot(xs, lst_sum_begin, "-", marker=".", color=colors[1])[0])
    plts.append(plt.plot(xs, lst_sum_begin_max, "-", marker=".", color=colors[2])[0])

    plt.legend(plts, legend)

    # plts = []
    # legend = ["sum min for p", "p = {}".format(p), "sum max for p"]
    # plt.figure()

    # plt.title("Sum, p = {}, watch min max values!".format(p))

    # plts.append(plt.plot(xs, lst_sum_min, "-", marker=".", color=colors[0])[0])
    # # plts.append(plt.plot(xs, lst_sum, "-", marker=".", color=colors[1])[0])
    # plts.append(plt.plot(xs, lst_sum_max, "-", marker=".", color=colors[2])[0])

    # plt.legend(plts, legend)

    plt.show()

    sys.exit()

    # colors = [
    #     "#0000FF",
    #     "#0022FF",
    #     "#0044FF",
    #     "#0066FF",
    #     "#0088FF",
    # ]
    colors = [get_random_color() for _ in range(0, len(lsts))]

    plt.figure()
    plt.title("Sum from beginning!")
    plts = []
    legends = []
    for idx, p, lst_sum_begin, _ in lsts:
    # for idx in range(0, 5):
        # ys = np.array(get_lst_from_dict(objs[(idx, 30)]))
        ys = np.array(lst_sum_begin)
        # ys = ys[1:]/ys[:-1]
        
        # ys = np.array([Dec(np.log(float(y))) for y in lst_sum_begin])
        # ys = np.array([log_dec(y) for y in ys])
        xs = np.arange(0, ys.shape[0])
        plts.append(plt.plot(xs, ys, "-", color=colors[idx], marker=".")[0])
        legends.append("p = {}".format(str(p)))
    plt.legend(plts, legends)
    # plt.plot(xs, np.arccos(1-1/2**xs)/np.arccos(0), "r.-")

    # plt.show()

    plt.figure()
    plt.title("Sum from 2**i to 2**(i+1)-1!")
    plts = []
    legends = []
    for idx, p, _, lst_sum in lsts:
    # for idx in range(0, 5):
        # ys = np.array(get_lst_from_dict(objs[(idx, 30)]))
        ys = np.array(lst_sum)
        # ys = ys[1:]/ys[:-1]
        ys = np.array([Dec(np.log(float(y))) for y in lst_sum])
        # ys = np.array([log_dec(y) for y in ys])
        xs = np.arange(0, ys.shape[0])
        plts.append(plt.plot(xs, ys, "-", color=colors[idx], marker=".")[0])
        legends.append("p = {}".format(str(p)))
    plt.legend(plts, legends)
    # plt.plot(xs, np.arccos(1-1/2**xs)/np.arccos(0), "r.-")

    plt.show()

    # y = np.array(lst)

    arr_begin = np.array([l[2] for l in lsts])

"""
sums: n_begin=0, n_end=15
[(0, Decimal('1.6449035488044354122131050571582721846699417190618')),
 (1, Decimal('2.6013267209351084066207512723802518931512757049650')),
 (2, Decimal('4.2978089129687701312020419380981211618256375808023')),
 (3, Decimal('6.4052216691993905763498655950846145493814157079695')),
 (4, Decimal('8.2275494200159700403376720249886426577329227049778')),
 (5, Decimal('9.4565824805509832528789143906237278256484020688498')),
 (6, Decimal('10.174935938439656054270960614782738325232294801966')),
 (7, Decimal('10.563909761032464373119409288882289763311705204673')),
 (8, Decimal('10.766386091417825286398875861106947972132917324468')),
 (9, Decimal('10.869693312186272484888810335805809976839423216278')),
 (10, Decimal('10.921873424375428657930178522183471526778630927831'))
]

sums: n_begin=0, n_end=18
[(0, Decimal('1.6449302521436848446063732022266601587532833238006')),
 (1, Decimal('2.6084690949601944921729747124625488997170940674029')),
 (2, Decimal('4.4183350462525424871855329863034421296947523144771')),
 (3, Decimal('6.9044480630322024245594230139911276726400032322171')),
 (4, Decimal('9.2457144264938567395447815559133774226972246446734')),
 (5, Decimal('10.911396375586244097387085821379185140371797537170')),
 (6, Decimal('11.914175357838909229129188940370545556155735610469')),
 (7, Decimal('12.465643280572762656569352843067595981839051055514')),
 (8, Decimal('12.754990458563266154874082246098053492149377516115')),
 (9, Decimal('12.903214248301431009146526995101918294121245789436')),
 (10, Decimal('12.978232733410339774963781609811201332945947264846'))
]

sums:
[(0, Decimal('1.6449338284296189131982260081125789023564638413670')),
 (1, Decimal('2.6113987861272806789656532087439237023007878713752')),
 (2, Decimal('4.5067234755604467773633572840989438432079016605338')),
 (3, Decimal('7.3970341617872865275461082709164963974700292863451')),
 (4, Decimal('10.412904299061791952243101577441439369206014805170'))
]

"""

#! /usr/bin/python2.7

# -*- coding: utf-8 -*-

import itertools
import primefac
import sys

import numpy as np

from collections import OrderedDict

sys.path.append("../combinatorics")
from different_combinations import get_permutation_table

# n = int(sys.argv[1])


def do_factor_stuff(n_old, recursion=0):
    i = 0
    if n_old > 10**8:
        return
    
    print("n_old: {}".format(n_old))
    if not n_old in n_map:
        n_map[n_old] = []

    factors = list(primefac.primefac(n_old))

    if len(factors) == 1:
        n_new = factors[0]
        print("finished here!")
        if not n_new in p_map:
            p_map[n_new] = 0
        return
    else:
        permutation_table = get_permutation_table(len(factors))
        # permutation_table = get_permutation_table(len(factors))[:3]
        factors_arr_str = np.array(list(map(str, factors)))
        lst = n_map[n_old]
        for row in permutation_table:
            n_new = int("".join(factors_arr_str[row]))
            if not n_new in lst:
                lst.append(n_new)
            if not n_new in n_map and recursion < 21:
                do_factor_stuff(n_new, recursion+1)


if __name__ == "__main__":
    n_map = {}
    p_map = {}

    for n in range(2, 16):
        if n%5 == 0:
            do_factor_stuff(n)

    n_map_sorted = OrderedDict(sorted(n_map.items()))

    # print("n_map_sorted:\n{}".format(n_map_sorted))

    for key, val in n_map_sorted.items():
        print("key: {}, val: {}".format(key, val))

    primes = sorted(list(p_map.keys()))
    print("primes:\n{}".format(primes))

     # a -> b [ label="a to b" ];
     # b -> c [ label="another label"];
    graph_str = """digraph G {
    rankdir=LR;  //left to right
    {}
    }"""

    # combine all keys and values from n_map and create an label map
    lst_vals = list(n_map.values())
    print("lst_vals:\n{}".format(lst_vals))

    nums_lst = list(itertools.chain.from_iterable(lst_vals))
    print("nums_lst:\n{}".format(nums_lst))

    nums_lst = list(set(list(n_map.keys())+nums_lst+list(p_map.keys())))

    # nums_lst_unique = list(set(nums_lst))
    # print("nums_lst_unique:\n{}".format(nums_lst_unique))
    # sys.exit(-123)

    # numbers = list(set(list(n_map.keys())+list(n_map.values())))
    # print("numbers:\n{}".format(numbers))

    num_node_label_map = {num: "x{}".format(i) for i, num in enumerate(nums_lst)}
    # num_node_label_map = {num: "x{}".format(i) for i, num in enumerate(numbers)}
    # print("num_node_label_map:\n{}".format(num_node_label_map))

    nodes = ""
    for key, val in num_node_label_map.items():
        nodes += "{} [label=\"{}\"];\n".format(val, key)

    relations = ""
    for key, val in n_map.items():
        if len(val) < 1:
            continue

        print("key: {}".format(key))
        print("val: {}".format(val))
        relations += "{} -> {{{}}};\n".format(num_node_label_map[key], ", ".join([num_node_label_map[v] for v in val]))

    # graph_str = graph_str.replace("{}", nodes)
    graph_str = graph_str.replace("{}", nodes+relations)

    with open("graph.dot", "w") as fout:
        fout.write(graph_str)

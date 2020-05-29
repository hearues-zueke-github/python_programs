#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy

import numpy as np

sys.path.append("../combinatorics/")
import different_combinations as combinations
sys.path.append("../math_numbers/")
from prime_numbers_fun import get_primes
sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj
import utils_all

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

if __name__ == "__main__":
    print("Hello World!")

    def get_f(file_extension_name):
        def f_norm(n):
            if n%2==0:
                return n//2
            else:
                return 3*n+1

        def f_ext(n):
            if n%2==0:
                return n//2
            elif n%3==0:
                return n//3
            else:
                return 3*n+1

        def f_ext2(n):
            if n%2==0:
                return n//2
            elif n%3==0:
                return n//3
            elif n%5==0:
                return n//5
            else:
                return 3*n+1

        def f_ext3(n):
            if n%3==0:
                return n//3
            elif n%5==0:
                return n//5
            elif n%7==0:
                return n//7
            elif n%2==0:
                return n//2
            else:
                return 4*n+1#+5*n**2

        def f_ext4(n):
            if n%7==0:
                return n//7
            elif n%7==1:
                return 3*n+1
            elif n%7==2:
                return 4*n+1
            elif n%7==3:
                return 2*n+1
            else:
                return 5*n+1#+5*n**2

        if file_extension_name=='normal_f':
            return f_norm
        elif file_extension_name=='extension_f':
            return f_ext
        elif file_extension_name=='extension2_f':
            return f_ext2
        elif file_extension_name=='extension3_f':
            return f_ext3
        elif file_extension_name=='extension4_f':
            return f_ext4

        assert False

    # file_extension_name = 'normal_f'
    file_extension_name = 'extension3_f'

    f = get_f(file_extension_name)

    n_max = 50000
    l = [(i, f(i)) for i in range(2, n_max+1)]
    print("l: {}".format(l))
    dl = {i: fi for i, fi in l}
    # remove every node, which has no connection with the root 1 so far!
    d1 = {i: [] for i in range(1, n_max+1)}
    for i1, i2 in l:
        if not i2 in d1:
            continue
        d1[i2].append(i1)

    for n_first, l_v in d1.items():
        if n_first==1:
            continue

        if len(l_v)>0:
            break

    s_complete = set([n_first])
    s_now = set([n_first])
    while True:
        s_next = set()
        for n in s_now:
            for v in d1[n]:
                if not v in s_complete:
                    s_next.add(v)

        if len(s_next)==0:
            break

        for n in s_next:
            s_complete.add(n)

        s_now = s_next

    # s_complete = set()
    # for n, lst_v in d1.items():
    #     has_on_predecessor = False
    #     for v in lst_v:
    #         if v in d1 and len(d1[v])>0:
    #             has_on_predecessor = True
    #             break
    #     if has_on_predecessor:
    #         # for v in lst_v:
    #         #     s_complete.add(v)
    #         s_complete.add(n)

    l_complete = [(i, dl[i]) for i in s_complete]

    a = np.array(l_complete).flatten()
    u, c = np.unique(a, return_counts=True)
    print("u: {}".format(u))
    print("c: {}".format(c))

    # create a simple digraph!

    file = open("collatz_graph_until_n_{n}_{file_extension_name}.gv".format(n=n_max, file_extension_name=file_extension_name), "w")

    file.write("digraph collatz_graph_until_n_{n} {{\n".format(n=n_max))

    node_to_node_name = {i: 'x{i}'.format(i=i) for i in u}
    for node, node_name in node_to_node_name.items():
        file.write("    {node_name}[label=\"{node}\"];\n".format(node_name=node_name, node=node))

    file.write("\n")

    for i1, i2 in l_complete:
        file.write("    {n1} -> {n2};\n".format(n1=node_to_node_name[i1], n2=node_to_node_name[i2]))
    file.write("\n")
    file.write("    labelloc=\"t\";\n")
    file.write("    label=\"n: {n}\";\n".format(n=n_max))
    file.write("}\n")
    file.close()


    s_complete = set()
    s_now = set([dl[n_first]])
    d_level = {1: [dl[n_first]]}
    level_nr = 2
    while True:
        s_next = set()
        for n in s_now:
            for v in d1[n]:
                if not v in s_complete:
                    s_next.add(v)

        if len(s_next)==0:
            break

        d_level[level_nr] = list(s_next)
        level_nr += 1

        for v in s_next:
            s_complete.add(v)

        s_now = s_next

    d_n_to_level = {}
    for level, l_v in d_level.items():
        for v in l_v:
            d_n_to_level[v] = level

    file = open("collatz_graph_until_n_{n}_numbering_{file_extension_name}.gv".format(n=n_max, file_extension_name=file_extension_name), "w")

    file.write("digraph collatz_graph_until_n_{n} {{\n".format(n=n_max))

    node_to_node_name = {i: 'x{i}'.format(i=i) for i in u}
    for node, node_name in node_to_node_name.items():
        file.write("    {node_name}[label=\"{node}\"];\n".format(node_name=node_name, node=d_n_to_level[node]))

    file.write("\n")

    for i1, i2 in l_complete:
        file.write("    {n1} -> {n2};\n".format(n1=node_to_node_name[i1], n2=node_to_node_name[i2]))
    file.write("\n")
    file.write("    labelloc=\"t\";\n")
    file.write("    label=\"n: {n}\";\n".format(n=n_max))
    file.write("}\n")
    file.close()

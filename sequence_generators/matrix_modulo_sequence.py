#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy

sys.path.append("../combinatorics/")
import different_combinations as combinations
sys.path.append("../graph_theory/")
from find_graph_cycles import get_cycles_of_1_directed_graph

import numpy as np

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

def pretty_str_list_1d(lst):
    return "[\n{}\n]".format(",\n".join(map(lambda x: "  "+str(x), lst)))


def generate_number_sequence(m, A, x):
    x = x.copy()
    multiply = m**np.arange(0, x.shape[0])

    num = np.sum(x*multiply)
    numbers = [num]

    x = (A.dot(x)) % m
    num = np.sum(x*multiply)
    while not num in numbers:
        numbers.append(num)
        x = (A.dot(x)) % m
        num = np.sum(x*multiply)

    return numbers


def get_numbers_table(m, n):
    numbers_table = np.zeros((m, m), dtype=np.int)
    multiply = m**np.arange(0, n)

    # lens = []
    print("m: {}, n: {}".format(m, n))

    for i, A in enumerate(combinations.get_all_combinations_repeat_generator(m, n**2), 0):
        if i % 100 == 0:
            print("i: {}".format(i))
        A = A.reshape((n, n))
        for j, x in enumerate(combinations.get_all_combinations_repeat_generator(m, n), 0):
            n1 = np.sum(x*multiply) % m
            n2 = np.sum(A.dot(x)*multiply) % m
            numbers_table[n1, n2] += 1

    return numbers_table


def find_other_factors():
    factors_lst = []
    max_factors = []
    lens = []

    ps = [2, 3, 5, 7, 11, 13, 17, 19]
    # m = int(sys.argv[1])
    # n = int(sys.argv[2])
    n = 2
    for m in range(1, 6):
        numbers_table = get_numbers_table(m, n)
        print("numbers_table:\n{}".format(numbers_table))

        uniques = np.unique(numbers_table)
        
        uniques_copy = uniques.copy()
        factors = []

        for p in ps:
            while np.all((uniques_copy % p)==0):
                factors.append(p)
                uniques_copy //= p
        # print("uniques_copy: {}".format(uniques_copy))

        factors_lst.append(factors)
        max_factors.append(np.max(uniques_copy))
        lens.append(len(uniques))
        print("factors: {}, uniques_copy: {}, lens: {}".format(factors, uniques_copy, lens))

    print("factors_lst: {}".format(factors_lst))
    print("max_factors: {}".format(max_factors))
    print("lens: {}".format(lens))

    """
    factors_lst: [[], [2, 2, 2], [3, 3, 3], [2, 2, 2, 2, 2, 2, 2], [5, 5, 5], [2, 2, 2, 3, 3, 3], [7, 7, 7], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3, 3], [2, 2, 2, 5, 5, 5], [11, 11, 11], [2, 2, 2, 2, 2, 2, 2, 3, 3, 3]]
    facotrs_multiplied: [1, 8, 27, 128, 125, 216, 343, 2048, 2187, 1000, 1331, 3456]
    max_factors: [1, 3, 5, 4, 9, 15, 13, 5, 7, 27, 21, 20]
    lens: [1, 3, 3, 4, 3, 8, 3, 5, 5, 9, 3, 11]
    """

def get_g(m, n):
    return combinations.get_all_combinations_repeat_generator(m, n)


def get_random_values(m, n):
    A = np.random.randint(0, m, (n, n))
    b = np.random.randint(0, m, (n, ))
    c = np.random.randint(0, m, (n, ))
    return A, b, c


def get_random_A(m, n):
    return np.random.randint(0, m, (n, n))


def convert_tpl_to_A_b_c(tpl, n):
    A = np.array(tpl[0:n**2]).reshape((n, n))
    b = np.array(tpl[n**2:n*(n+1)]).reshape((n, ))
    c = np.array(tpl[n*(n+1):n*(n+2)]).reshape((n, ))
    return A, b, c


def do_cycles(A, b, c, m):
    lst = [b]
    x_prev = b.copy()
    x = (A.dot(x_prev)+c) % m
    x_tpl = tuple(x)
    set_tpl = {tuple(b)}

    while not x_tpl in set_tpl:
        lst.append(x)
        set_tpl.add(x_tpl)
        x_prev = x
        x = (A.dot(x_prev)+c) % m
        x_tpl = tuple(x)

    return lst


def find_best_random_A_b_c(m, n):
    all_combinations = list(get_g(m, n))
    all_combinations = set(map(tuple, all_combinations))

    b = np.zeros((n, ), dtype=np.int)
    c = np.zeros((n, ), dtype=np.int)
    c[0] = 1

    while True:
        A = get_random_A(m, n)

        lst = do_cycles(A, b, c, m)
        set_cyclics = set(map(tuple, lst))

        non_cyclic_lst = list(all_combinations-set_cyclics)
        if len(non_cyclic_lst) == 1:
            break

    return A, b, c, np.array(non_cyclic_lst[0])


def find_all_best_A(m, n):
    all_combinations = list(get_g(m, n))
    all_combinations = set(map(tuple, all_combinations))

    # As = []

    b = np.zeros((n, ), dtype=np.int)
    c = np.zeros((n, ), dtype=np.int)
    c[0] = 1

    for a in get_g(m, n**2):
        A = a.reshape((n, n))

        lst = do_cycles(A, b, c, m)
        set_cyclics = set(map(tuple, lst))

        non_cyclic_lst = list(all_combinations-set_cyclics)
        if len(non_cyclic_lst) == 1:
            # As.append(A)
            yield A

    # return As


def find_corresponding_non_cyclic_b_c(A, m, n):
    assert A.shape==(n, n)
    assert np.all((A<m)&(A>=0))

    all_combinations = list(get_g(m, n))
    all_combinations = set(map(tuple, all_combinations))

    # test if A is a valid matrix for max cyclic length!
    b = np.zeros((n, ), dtype=np.int)
    c = np.zeros((n, ), dtype=np.int)
    c[0] = 1

    def get_non_cyclic_lst(A, b, c, m):
        lst = do_cycles(A, b, c, m)
        set_cyclics = set(map(tuple, lst))
        non_cyclic_lst = list(all_combinations-set_cyclics)
        return non_cyclic_lst

    non_cyclic_lst = get_non_cyclic_lst(A, b, c, m)
    assert len(non_cyclic_lst)==1

    c_gen = get_g(m, n)

    c = next(c_gen)
    b[0] = 1
    non_cyclic_lst = get_non_cyclic_lst(A, b, c, m)

    b[0] = 0
    corresponding_b_c_lst = []
    for c in c_gen:
        non_cyclic_lst = get_non_cyclic_lst(A, b, c, m)
        assert len(non_cyclic_lst)==1
        corresponding_b_c_lst.append((tuple(non_cyclic_lst[0]), tuple(c)))
    # print("A: {}".format(A))
    # print("corresponding_b_c_lst: {}".format(corresponding_b_c_lst))
    # input("ENTER3...")
    return corresponding_b_c_lst


def find_graph_loops_in_corresponding_b_c(corresponding_b_c_lst):
    c_to_b_map = {c: b for b, c in corresponding_b_c_lst}
    print("c_to_b_map:\n{}".format(c_to_b_map))

    c_lst = list(c_to_b_map.values())
    print("c_lst: {}".format(c_lst))


def create_c_to_b_graph_image(A, corresponding_b_c_lst, m, n):
    b_lst, c_lst = list(zip(*corresponding_b_c_lst))

    b_arr = np.array(b_lst, dtype=np.int64)
    c_arr = np.array(c_lst, dtype=np.int64)

    unique_c, unique_c_count = np.unique(c_arr.reshape((-1, )).view("i8,i8"), return_counts=True)
    # test if each c is comming only once!
    # for beeing sure for getting only a graph with one input many output!
    assert np.all(unique_c_count==1)

    # print("b_arr:\n{}".format(b_arr))
    # print("c_arr:\n{}".format(c_arr))

    # print("b_arr.dtype: {}".format(b_arr.dtype))
    # print("c_arr.dtype: {}".format(c_arr.dtype))

    # print("b_lst:\n{}".format(b_lst))
    # print("c_lst:\n{}".format(c_lst))

    c_to_b_map = {c: b for b, c in corresponding_b_c_lst}
    print("c_to_b_map:\n{}".format(c_to_b_map))

    for c, b in c_to_b_map.items():
        print("c: {}, b: {}".format(c, b))

    node_to_node_name = {}
    node_nr = 0
    for c in c_to_b_map.keys():
        if not c in node_to_node_name:
            node_to_node_name[c] = "x{}".format(node_nr)
            node_nr += 1
    for b in c_to_b_map.values():
        if not b in node_to_node_name:
            node_to_node_name[b] = "x{}".format(node_nr)
            node_nr += 1

    print("node_to_node_name: {}".format(node_to_node_name))

    file = open("graph_m_{}_n_{}_A_{}_c_to_b.gv".format(m, n, A.tolist()), "w")

    file.write("digraph c_to_b_vector_correlations {\n")

    for node, node_name in node_to_node_name.items():
        file.write("    {node_name}[label=\"{node}\"];\n".format(node_name=node_name, node=node))

    file.write("\n")

    for c, b in c_to_b_map.items():
        file.write("    {n1} -> {n2};\n".format(n1=node_to_node_name[c], n2=node_to_node_name[b]))

    file.write("\n")

    file.write("    labelloc=\"t\";\n")
    file.write("    label=\"m: {}, n: {}\nA: {}\";\n".format(m, n, A.tolist()))

    file.write("}\n")

    file.close()

    # c_lst = list(c_to_b_map.values())
    # print("c_lst: {}".format(c_lst))
    return c_to_b_map


def create_cyclic_graphs(m, n):
    A_best, b, c, b_non_cyclic = find_best_random_A_b_c(m, n)
    # A_best = np.array([[0, 2], [2, 2]])
    print("A_best.tolist(): {}".format(A_best.tolist()))
    print("b.tolist(): {}".format(b.tolist()))
    print("c.tolist(): {}".format(c.tolist()))
    print("b_non_cyclic.tolist(): {}".format(b_non_cyclic.tolist()))

    corresponding_b_c_lst = find_corresponding_non_cyclic_b_c(A_best, m, n)

    print("corresponding_b_c_lst: {}".format(corresponding_b_c_lst))

    # find_graph_loops_in_corresponding_b_c(corresponding_b_c_lst)

    # create a corresponding_b_c network graph!
    c_to_b_map = create_c_to_b_graph_image(A_best, corresponding_b_c_lst, m, n)


def analyse_different_b_c_cycles(m, n):
    # As = find_all_best_A(m, n)
    # A_best = np.array([[0, 2], [2, 2]])
    
    lst_of_A_of_c_b_cycles = []
    for A_best in find_all_best_A(m, n):
        print("A_best.tolist(): {}".format(A_best.tolist()))
        # print("b.tolist(): {}".format(b.tolist()))
        # print("c.tolist(): {}".format(c.tolist()))
        # print("b_non_cyclic.tolist(): {}".format(b_non_cyclic.tolist()))

        corresponding_b_c_lst = find_corresponding_non_cyclic_b_c(A_best, m, n)

        # print("corresponding_b_c_lst: {}".format(corresponding_b_c_lst))

        nodes_sorted = sorted(set([n for l in corresponding_b_c_lst for n in l]))
        mapping_dict = {n: i for i, n in enumerate(nodes_sorted, 0)}
        mapping_dict_rev = {i: n for n, i in mapping_dict.items()}
        edges_directed = [(mapping_dict[c], mapping_dict[b]) for b, c in corresponding_b_c_lst]
        print("edges_directed: {}".format(edges_directed))
        list_of_cycles = get_cycles_of_1_directed_graph(edges_directed)
        print("list_of_cycles: {}".format(list_of_cycles))
        lst_of_c_b_cycles = [[mapping_dict_rev[i] for i in cycles] for cycles in list_of_cycles]
        print("lst_of_c_b_cycles: {}".format(lst_of_c_b_cycles))
        print("len(lst_of_c_b_cycles): {}".format(len(lst_of_c_b_cycles)))

        lst_of_A_of_c_b_cycles.append((tuple(A_best.reshape((-1, ))), lst_of_c_b_cycles))

        # input("ENTER...")

    lst_of_A_of_c_b_cycles = sorted(lst_of_A_of_c_b_cycles, key=lambda x: (len(x[1]), x[0]))

    lst_of_A_of_c_b_cylces_expand = list(map(lambda x: (x[0], list(map(lambda y: y+y[:1], x[1]))),lst_of_A_of_c_b_cycles))
    print("lst_of_A_of_c_b_cylces_expand: {}".format(lst_of_A_of_c_b_cylces_expand))

    dict_corresponding_c_b_equals = {}
    for i, (A1, cycles1) in enumerate(lst_of_A_of_c_b_cylces_expand, 0):
        for j, (A2, cycles2) in enumerate(lst_of_A_of_c_b_cylces_expand[i+1:], i+1):
            # print("i: {}, j: {}".format(i, j))
            print("A1: {}, A2: {}".format(A1, A2))
            for cycle1 in cycles1:
                for cycle2 in cycles2:
                    for c1, b1 in zip(cycle1[:-1], cycle1[1:]):
                        for c2, b2 in zip(cycle2[:-1], cycle2[1:]):
                            if c1==c2 and b1==b2:
                                t = (c1, b1)
                                if not t in dict_corresponding_c_b_equals:
                                    dict_corresponding_c_b_equals[t] = [A1, A2]
                                d = dict_corresponding_c_b_equals[t]
                                if not A1 in d:
                                    d.append(A1)
                                if not A2 in d:
                                    d.append(A2)
    print("dict_corresponding_c_b_equals: {}".format(dict_corresponding_c_b_equals))

    corresponding_A_of_c_b_lens = [(k, len(v)) for k, v in dict_corresponding_c_b_equals.items()]
    corresponding_A_of_c_b_lens = sorted(corresponding_A_of_c_b_lens, key=lambda x: (x[1], x[0]))
    print("corresponding_A_of_c_b_lens: {}".format(corresponding_A_of_c_b_lens))

    corresponding_A_of_c_b = [t for t, _ in corresponding_A_of_c_b_lens]
    print("corresponding_A_of_c_b: {}".format(corresponding_A_of_c_b))

    # try only one c, b pair to find the reverse cyclic
    max_equal_positions = 0
    best_A1 = 0
    best_A2 = 0
    best_b = 0
    best_c = 0
    for c, b in corresponding_A_of_c_b:
        # c, b = corresponding_A_of_c_b[0]
        print("c: {}, b: {}".format(c, b))

        lst_A = dict_corresponding_c_b_equals[(c, b)]
        # print("lst_A: {}".format(lst_A))

        lst_A_arr = [np.array(A).reshape((n, n)) for A in lst_A]

        b = np.array(b)
        c = np.array(c)

        b[0] = (b[0]+1) % m

        # print("b: {}".format(b))
        # print("c: {}".format(c))

        # for i, A in enumerate(lst_A_arr, 0):
        #     print("i: {}, A: {}".format(i, A))
        #     cycle = do_cycles(A, b, c, m)
        #     print("cycle: {}".format(cycle))
        #     print("len(cycle): {}".format(len(cycle)))
        #     input("ENTER...")

        for i, A1 in enumerate(lst_A_arr, 0):
            cycle_1 = do_cycles(A1, b, c, m)
            cycle_1_tpl = list(map(tuple, cycle_1))
            cycle_1_arr = np.array(cycle_1_tpl)
            for j, A2 in enumerate(lst_A_arr[i+1:], i+1):
                cycle_2 = do_cycles(A2, b, c, m)
                cycle_2_tpl = list(map(tuple, cycle_2))
                cycle_2_tpl_inv = cycle_2_tpl[:1]+cycle_2_tpl[1:][::-1]
                cycle_2_arr = np.array(cycle_2_tpl_inv)


                equal_positions = np.sum(np.all(cycle_1_arr==cycle_2_arr, axis=1))
                if max_equal_positions < equal_positions:
                    max_equal_positions = equal_positions
                    best_A1 = A1
                    best_A2 = A2
                    best_b = b
                    best_c = c
                # print("cycle_1_tpl: {}".format(cycle_1_tpl))
                # print("cycle_2_tpl: {}".format(cycle_2_tpl))
                # print("cycle_2_tpl_inv: {}".format(cycle_2_tpl_inv))
                
                # print("equal_positions: {}".format(equal_positions))

    print("max_equal_positions: {}".format(max_equal_positions))
    print("best_A1: {}, best_A2: {}".format(best_A1, best_A2))
    print("best_b: {}, best_c: {}".format(best_b, best_c))
    # input("ENTER...")

    return lst_of_A_of_c_b_cycles


if __name__ == "__main__":
    def get_2d_generator(f_get_gen1, f_get_gen2, args1=(), args2=()):
        for i in f_get_gen1(*args1):
            for j in f_get_gen2(*args2):
                yield (i, j)

    def get_3d_generator(f_get_gen1, f_get_gen2, f_get_gen3, args1=(), args2=(), args3=()):
        for i in f_get_gen1(*args1):
            for j in f_get_gen2(*args2):
                for k in f_get_gen3(*args3):
                    yield (i, j, k)

    def get_own_2d_generator(f_get_gen, args=()):
        for i in f_get_gen(*args):
            for j in f_get_gen(*args):
                yield (i, j)

    def get_r1():
        return range(0, 3)
    def get_r2():
        return range(5, 7)

    lst1 = [(i, j) for i in get_r1() for j in get_r2()]
    lst2 = list(get_2d_generator(get_r1, get_r2))
    assert lst1 == lst2

    lst1 = [(i, j) for i in get_r1() for j in get_r1()]
    lst2 = list(get_2d_generator(get_r1, get_r1))
    assert lst1 == lst2

    m = int(sys.argv[1])
    n = int(sys.argv[2])
    print("m: {}, n: {}".format(m, n))

    # create_cyclic_graphs(m, n)
    lst_of_A_of_c_b_cycles = analyse_different_b_c_cycles(m, n)
    sys.exit(0)


    # m = 2
    # n = 7
    # def get_numbers_analysis_values(m, n):
    #     # first approach to find numbers_table and counts etc.
    #     print("m: {}, n: {}".format(m, n))

    #     numbers_count = np.zeros(((m-1)**2*n+1, ), dtype=np.int)
    #     numbers_count_mod = np.zeros((m, ), dtype=np.int)
    #     numbers_table = np.zeros((m**n, m**n), dtype=np.int)
    #     multiply = m**np.arange(0, n)

    #     i = -1
    #     # for a1, a2 in get_2d_generator(get_g, get_g, (m, n), (m, n)):
    #     for a1, a2 in get_own_2d_generator(get_g, (m, n)):
    #         num = a1.dot(a2)
    #         numbers_count[num] += 1
    #         numbers_count_mod[num % m] += 1
    #         numbers_table[np.sum(a1*multiply), np.sum(a2*multiply)] = num
    #     print("numbers_count: {}".format(numbers_count))
    #     print("numbers_count_mod: {}".format(numbers_count_mod))
    #     print("numbers_table:\n{}".format(numbers_table))

    #     return numbers_count, numbers_count_mod, numbers_table


    # first approach to find best Matrix A and Vector b for getting longest cycle!
    m = int(sys.argv[1])
    n = int(sys.argv[2])
    # n = 4 # stay simple with n==2 !!!

    amount_unique = 0
    amount_found_again = 0
    lens_values = {}
    lens_unique_count = {}
    best_factors = (0, 0, 0)
    best_factors_lst = []

    max_len = 0

    max_lens_dict = {
        (2, 2): 4, (3, 2): 8, (4, 2): 8, (5, 2): 24,
        (6, 2): 24, (7, 2): 48, (8, 2): 16, (9, 2): 24, (10, 2): 60,
    }

    t = (m, n)
    if t in max_lens_dict:
        max_len = max_lens_dict[t]
        print("max_len: {}".format(max_len))

    b = np.zeros((n, ), dtype=np.int)
    c = np.zeros((n, ), dtype=np.int)
    c[0] = 1

    # Monte Carlo approach!
    for i in range(0, 2200):
        if i % 10000 == 0:
            print("i: {}".format(i))
        # A, b, c = get_random_values(m, n)
        A = get_random_A(m, n)
        # b[:] = 0
        # c[:] = 0
        # c[0] = 1

        t = tuple(A.reshape((-1, )))+tuple(b)+tuple(c)
        if not t in lens_values:
            lst = do_cycles(A, b, c, m)
            lens_values[t] = 0
            l = len(lst)
            if not l in lens_unique_count:
                lens_unique_count[l] = 0
            lens_unique_count[l] += 1
            if max_len < l:
                best_factors = (A, b, c)
                best_factors_lst = [best_factors]
                max_len = l
                print("new max_len: {}".format(max_len))
            elif max_len == l:
                best_factors_lst.append((A, b, c))
            amount_unique += 1
        else:
            amount_found_again += 1

    lens_unique, unique_counts = list(zip(*list(lens_unique_count.items())))
    lens_unique = np.array(lens_unique)
    unique_counts = np.array(unique_counts)

    new_idxs = np.argsort(lens_unique)
    lens_unique = lens_unique[new_idxs].tolist()
    unique_counts = unique_counts[new_idxs].tolist()

    len_lens_unique = len(lens_unique)

    print("len(lens_values): {}".format(len(lens_values)))
    print("amount_unique: {}".format(amount_unique))
    print("amount_found_again: {}".format(amount_found_again))

    print("m: {}, n: {}".format(m, n))
    print("max_len: {}".format(max_len))
    print("len_lens_unique: {}".format(len_lens_unique))
    print("lens_unique: {}".format(lens_unique))
    print("unique_counts: {}".format(unique_counts))

    tpls_lst = list(map(lambda x: tuple(x[0].reshape((-1, )))+tuple(x[1])+tuple(x[2]), best_factors_lst))
    tpls_arr = np.array(tpls_lst)
    # swap b and c columns!
    tpls_arr = np.hstack((tpls_arr[:, 0:n*(n+0)], tpls_arr[:, n*(n+1):n*(n+2)], tpls_arr[:, n*(n+0):n*(n+1)]))

    # sort be columns from left to right
    tpls_arr = tpls_arr[np.argsort(tpls_arr.reshape((-1, )).view(",".join(['i8']*(n*(2+n)))))]

    arr_A = tpls_arr[:, :n**2]
    idxs_A = np.hstack(((0, ), np.where(np.any(arr_A[:-1]!=arr_A[1:], axis=1))[0]+1, (tpls_arr.shape[0], )))
    different_As = arr_A[idxs_A[:-1]]
    print("len(different_As): {}".format(len(different_As)))

    arr_Ab = np.hstack((tpls_arr[:, :n**2], tpls_arr[:, n*(n+0):n*(n+1)]))
    idxs_Ab = np.hstack(((0, ), np.where(np.any(arr_Ab[:-1]!=arr_Ab[1:], axis=1))[0]+1, (tpls_arr.shape[0], )))
    different_Ab = arr_Ab[idxs_Ab[:-1]]
    print("len(different_Ab): {}".format(len(different_Ab)))

    arr_A_only = different_Ab[:, :n**2]
    idxs_A_only = np.hstack(((0, ), np.where(np.any(arr_A_only[:-1]!=arr_A_only[1:], axis=1))[0]+1, (arr_A_only.shape[0], )))
    different_A_only = arr_A_only[idxs_A_only[:-1]]
    print("len(different_A_only): {}".format(len(different_A_only)))

    set_booleans = np.vstack([np.any(different_As==i, axis=1) for i in range(1, m)]).T
    lst_including_numbers = list(map(lambda x: tuple(np.where(x)[0]+1),set_booleans))
    set_including_numbers = set(lst_including_numbers)
    sorted_sets = sorted(list(set_including_numbers), key=lambda x: (len(x), x))

    counting_sets = [(s, lst_including_numbers.count(s)) for s in sorted_sets]
    print("counting_sets: {}".format(pretty_str_list_1d(counting_sets)))

    # test only one cycle, to find which one x is not found in the lst_cylce!
    A, b, c = convert_tpl_to_A_b_c(tpls_lst[0], n)
    lst_cycle = do_cycles(A, b, c, m)
    set_lst_cycle = set(map(tuple, lst_cycle))
    print("set_lst_cycle: {}".format(set_lst_cycle))

    all_combinations = list(get_g(m, n))
    all_combinations = set(map(tuple, all_combinations))

    non_cyclic_elements = list(all_combinations-set_lst_cycle)
    print("non_cyclic_elements: {}".format(non_cyclic_elements))

    # test if the found non_cyclic b is really non_cyclic!
    b_non_cyclic = np.array(non_cyclic_elements[0])

    print("A: {}".format(A.tolist()))
    print("b_non_cyclic: {}".format(b_non_cyclic.tolist()))
    print("c: {}".format(c.tolist()))

    lst_non_cyclic = do_cycles(A, b_non_cyclic, c, m)
    assert len(lst_non_cyclic)
    b_non_cylic_2 = lst_non_cyclic[0]
    print("b_non_cylic_2: {}".format(b_non_cylic_2.tolist()))
    assert np.all(b_non_cyclic==b_non_cylic_2)

    idxs_lower_values = np.where((idxs_A_only[1:]-idxs_A_only[:-1])==m**2-1)[0]



    sys.exit(0)

    lens = list(map(len, list(lens_values.values())))
    uniques, uniques_count = np.unique(lens, return_counts=True)
    print("m: {}, n: {}".format(m, n))
    print("uniques:\n{}".format(uniques))
    print("uniques_count:\n{}".format(uniques_count))

    # lst = do_cycles(A, b, c, m)
    # print("lst: {}".format(lst))

    # possible intersting sequences for the oeis.org site!
    # max cycles per modulo for n=2
    # [1,4,8,8,24,24,48,16,24,60,120,168,84,...]
    # len of different cycles lens per modulo for n=2
    # [1,4,6,7,11,13,14,13,14,24,20,...]
    # max different A's matrices per modulo for n=2
    # [1,3,12,24,80,24,336,384,864,216,1760,288,...]

    sys.exit(-1)

    default_vector_cycle_table = {tuple(i): [] for i in get_g(m, n)}
    vector_vector_cycle_table = {tuple(i): deepcopy(default_vector_cycle_table) for i in get_g(m, n)}
    matrix_vector_vector_forward_cycle_table = {tuple(i): deepcopy(vector_vector_cycle_table) for i in get_g(m, n**2)}
    matrix_vector_vector_backward_cycle_table = {tuple(i): deepcopy(vector_vector_cycle_table) for i in get_g(m, n**2)}

    lens_table = {tuple(i1)+tuple(i2)+tuple(i3): 0 for i1, i2, i3 in get_3d_generator(get_g, get_g, get_g, (m, n**2), (m, n), (m, n))}

    for a1, a2, a3 in get_3d_generator(get_g, get_g, get_g, (m, n**2), (m, n), (m, n)):
    # for a1, a2 in get_2d_generator(get_g, get_g, (m, n**2), (m, n)):
        print("a1: {}, a2: {}, a3: {}".format(a1, a2, a3))
        a1_tup = tuple(a1) # A
        a2_tup = tuple(a2) # c

        A = a1.reshape((n, n))
        c = a2

        b_start = a3 # b
        b = b_start.copy()

        b_next = (A.dot(b)+c) % m
        l = 1
        if np.any(b_start!=b_next):
            while np.any(b!=b_next):
                b_tpl = tuple(b)
                b_next_tpl = tuple(b_next)
                matrix_vector_vector_forward_cycle_table[a1_tup][a2_tup][b_tpl].append(b_next_tpl)
                matrix_vector_vector_backward_cycle_table[a1_tup][a2_tup][b_next_tpl].append(b_tpl)
                temp = b_next
                b_next = (A.dot(b)+c) % m
                b = temp
                l += 1
        else:
            b_tpl = tuple(b)
            matrix_vector_vector_forward_cycle_table[a1_tup][a2_tup][b_tpl].append(b_tpl)
            matrix_vector_vector_backward_cycle_table[a1_tup][a2_tup][b_tpl].append(b_tpl)

        lens_table[a1_tup+a2_tup+tuple(b_start)] = l


    sys.exit(-1)

    ms = []
    lens_of_lens_unique = []
    max_lens = []

    for m in range(1, 4):
        # m = 1
        n = 2
        # arr = combinations.get_all_combinations_repeat(4, n**2)
        # print("arr:\n{}".format(arr))

        lens = []
        print("m: {}, n: {}".format(m, n))

        for i, A in enumerate(combinations.get_all_combinations_repeat_generator(m, n**2), 0):
        # for i, a in enumerate(combinations.get_all_combinations_repeat_generator(m, n), 0):
            if i % 100 == 0:
                print("i: {}".format(i))
            A = A.reshape((n, n))
            # print("i: {}, A:\n{}".format(i, A))
            for j, x in enumerate(combinations.get_all_combinations_repeat_generator(m, n), 0):
                # x = x_
                # print("  j: {}, x: {}".format(j, x))
                numbers = generate_number_sequence(m, A, x)
                # print("    numbers: {}".format(numbers))
                lens.append(len(numbers))

        lens_unique, lens_count = np.unique(lens, return_counts=True)

        """
        # for n == 1
            ms
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
            lens of lens unique
            [1, 2, 2, 3, 3, 3, 4, 4, 4]
            max lengths
            [1, 2, 2, 3, 4, 3, 6, 4, 6]
        # for n == 2
            # lens of the different lens of possible combinations!
            [1, 3, 6, 6, 11, 12, 14]
            # max length of the numbers list!
            [1, 3, 8, 6, 24, 24, 48]

        # for n == 3
            # ms =
            [1, 2, 3, 4]
            # lens of unique lens =
            [1, 5, 11, 9]
            # max length =
            [1, 7, 26, 14]
        """
        print("lens_unique:\n{}".format(lens_unique))
        print("lens_count:\n{}".format(lens_count))

        print("len(lens_unique): {}".format(len(lens_unique)))
        print("np.max(lens_unique): {}".format(np.max(lens_unique)))

        ms.append(m)
        lens_of_lens_unique.append(len(lens_unique))
        max_lens.append(np.max(lens_unique))

    for m, len_of_lens_unique, max_len in zip(ms, lens_of_lens_unique, max_lens):
        print("m: {}, len_of_lens_unique: {}, max_lens: {}".format(m, len_of_lens_unique, max_len))

    print("ms:\n{}".format(ms))
    print("lens_of_lens_unique:\n{}".format(lens_of_lens_unique))
    print("max_lens:\n{}".format(max_lens))

    # m = 5
    # modulo = 3

    # a = np.random.randint(0, 2, (m, ))
    # b = np.random.randint(0, 2, (m, ))

    # C = np.add.outer(a, b) % modulo

    # an1 = np.dot(C, a) % modulo
    # bn1 = np.dot(C, b) % modulo

    # an2 = np.dot(a, C) % modulo
    # bn2 = np.dot(b, C) % modulo

    # print("a:\n{}".format(a))
    # print("b:\n{}".format(b))
    # print("C:\n{}".format(C))
    # print("an1:\n{}".format(an1))
    # print("bn1:\n{}".format(bn1))
    # print("an2:\n{}".format(an2))
    # print("bn2:\n{}".format(bn2))

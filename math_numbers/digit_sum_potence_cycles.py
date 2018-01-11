#! /usr/bin/python3.5

import os
import sys
import time

import multiprocessing as mp
import numpy as np

from functools import reduce
from multiprocessing import Process, Queue
from subprocess import check_call

def get_all_combinations(length, b):
    def get_combinations(start, length, full_length, b):
        if length == 1:
            z = np.zeros((b-start, full_length)).astype(np.int)
            z[:, full_length-1] = np.arange(start, b)
            return z

        l = []
        for i in range(start, b):
            n = get_combinations(i, length-1, full_length, b)
            n[:, full_length-length] = i
            l.extend(n.tolist())
        l = np.array(l).astype(np.int)
        return np.array(l).astype(np.int)
    combos = get_combinations(0, length, length, b)
    multiplies = np.zeros(length)
    for i in range(0, length):
        multiplies[length-1-i] = b**i
    return np.sum(combos*multiplies, axis=1).astype(np.int).tolist()

def get_digit_list(n, b):
    l = []
    while n > 0:
        l.append(n % b)
        n //= b
    return l

def get_cycles(n, b, d):
    cycles = []
    while not n in cycles:
        cycles.append(n)
        n = np.sum(np.array(get_digit_list(n, b)).astype(np.int)**d)
    # for getting again the repeated cycles
    new_cycles = []
    while not n in new_cycles:
        new_cycles.append(n)
        n = np.sum(np.array(get_digit_list(n, b)).astype(np.int)**d)
    # return new_cycles
    cycles = cycles+new_cycles
    return cycles[cycles.index(cycles[-1]):]

num_to_str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#?"

def calc_num_to_base(i, b):
    if i == 0:
        return "0"
    l = []
    while i > 0:
        l.append(i%b)
        i //= b
    l = l[::-1]
    return "".join(map(lambda x: num_to_str[x], l))

def create_graph(transitions, file_name, base):
    with open(file_name, "w") as f:
        f.write("digraph graphname {\n")

        for i, _ in transitions:
            f.write("    a{} [label=\"{}\"];\n".format(calc_num_to_base(i, base), calc_num_to_base(i, base)))

        f.write("\n")

        for i, j in transitions:
            f.write("    a{} -> a{};\n".format(calc_num_to_base(i, base), calc_num_to_base(j, base)))

        f.write("}\n")

def get_cycles_graph(l, b, d):
    combos = get_all_combinations(l, b)
    all_cycles = []
    for i in combos: # range(0, n):
        all_cycles.append([i, get_cycles(i, b, d)])

    all_cycles = sorted(all_cycles, key=lambda x: len(x[1]))

    transitions = []
    for _, cycles in all_cycles:
        for i, j in zip(cycles[:-1], cycles[1:]):
            transitions.append((i, j))
    transitions_dups = sorted(transitions, key=lambda x: x[0])

    # remove duplicates
    transitions = []
    length = len(transitions_dups)
    for i in range(0, length):
        t = transitions_dups.pop(0)
        if not t in transitions:
            transitions.append(t)

    # print("transitions:\n{}".format(transitions))
    file_name = "cycle_b_{}_d_{}_l_{}_".format(b, d, l)
    file_name_dot = file_name+".dot"
    file_name_png = file_name+".png"
    create_graph(transitions, file_name_dot, b)
    check_call(['dot', '-Tpng', file_name_dot, '-o', file_name_png])
    check_call(["rm", file_name_dot])

def find_largest_needed_length(base, power):
    base_1 = base-1
    i = 1
    max_diff = 0
    while True:
    # for i in range(1, 20):
        x = calc_num_to_base(i*base_1**power, base)
        y = calc_num_to_base(base**i, base)
        length = len(x)
        if length < i:
            break
        else:
            diff = i - length
            if max_diff < diff:
                max_diff = diff
            # print("x: {}, y: {}".format(x, y))
        i += 1
    # print("max_diff: {}".format(max_diff))
    return i-1

def process_get_cycles_graph(l, b, d, proc_id, queue):
    get_cycles_graph(l, b, d)
    queue.put(proc_id)

def dummy(proc_id, queue):
    queue.put(proc_id)

if __name__ == "__main__":
    max_cpu = 8
    queue = Queue()
    procs = {(i, ): Process(target=dummy, args=((i, ), queue)) for i in range(0, max_cpu)}
    for proc_id in procs:
        procs[proc_id].start()

    test_output = list(map(lambda x: calc_num_to_base(x, 4), get_all_combinations(2, 4)))
    print("test_output: {}".format(test_output))

    os.chdir(os.environ['HOME'])
    os.chdir("Pictures")
    if not os.path.exists("cycles"):
        os.makedirs("cycles")
    os.chdir("cycles")


    # for l in range(2, 8):
    # print("l: {}".format(l))
    for b in range(30, 65):
    # for b in range(2, 37):
    # for b in range(2, 21):
        print("b: {}".format(b))
        for d in range(2, 3):
            l = find_largest_needed_length(b, d)
            print("  d: {}, l: {}".format(d, l))

            proc_id = queue.get()
            procs[proc_id].join()
            del procs[proc_id]

            print("finished proc_id: {}".format(proc_id))
            
            new_proc_id = (b, d)
            proc = Process(target=process_get_cycles_graph, args=(l, b, d, new_proc_id, queue))

            proc.start()
            procs[new_proc_id] = proc

    rest_procs = len(procs)
    for _ in range(0, rest_procs):
        proc_id = queue.get()
        procs[proc_id].join()
        del procs[proc_id]

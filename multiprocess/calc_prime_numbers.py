#! /usr/bin/python3

# -*- coding: utf-8 -*-

import marshal
import pickle

import multiprocessing as mp
import numpy as np

from multiprocessing import Process, Queue

from time import time

from multiprocess_template_program import ProcessPool

def setup_primes(functions, variables, ps):
    variables["ps"] = ps

def calc_primes(functions, variables, n_start, n_end):
    ps = variables["ps"]
    proc_num = variables["proc_num"]

    if n_start % 2 == 0:
        n_start += 1
    if n_start % 3 == 0:
        n_start += 2

    diffs = np.array([0, 2])
    if (n_start+2) % 3 == 0:
        diffs = np.array([0, 4])

    n_diff = n_end-n_start

    pseudo_numbers = np.arange(0, (lambda x: x+2 if x*6+n_start+diffs[1]<n_end else x+1)((n_diff+diffs[1])//6))*6+n_start
    pseudo_primes = np.add.outer(pseudo_numbers, diffs).reshape((-1, ))

    max_primes = int(np.sqrt(n_end))+1
    ps_choosen = ps[ps<max_primes]
    print("ps_choosen:\n{}".format(ps_choosen))

    primes = pseudo_primes[~(np.sum(np.mod.outer(pseudo_primes, ps_choosen)==0, axis=1) > 0)]

    print("primes: {}".format(primes))

    return primes[primes<n_end]
    # return primes[:10]


def calc_primes_standard(ps, n_start, n_end):
    if n_start % 2 == 0:
        n_start += 1
    if n_start % 3 == 0:
        n_start += 2

    diffs = np.array([0, 2])
    if (n_start+2) % 3 == 0:
        diffs = np.array([0, 4])

    n_diff = n_end-n_start

    pseudo_numbers = np.arange(0, (lambda x: x+2 if x*6+n_start+diffs[1]<n_end else x+1)((n_diff+diffs[1])//6))*6+n_start
    pseudo_primes = np.add.outer(pseudo_numbers, diffs).reshape((-1, ))

    max_primes = int(np.sqrt(n_end))+1
    ps_choosen = ps[ps<max_primes]
    print("ps_choosen.shape:\n{}".format(ps_choosen.shape))

    primes = pseudo_primes[~(np.sum(np.mod.outer(pseudo_primes, ps_choosen)==0, axis=1) > 0)]

    print("primes.shape: {}".format(primes.shape))

    return primes

if __name__ == "__main__":
    ps = np.array([2, 3, 5, 7])

    numbers_range = np.array([8, 45, 1000, 10000, 100000, 1000000])

    start_time = time()
    for i in range(0, len(numbers_range)-1):
        print("i: {}".format(i))
        primes = calc_primes_standard(ps, numbers_range[i]+1, numbers_range[i+1])
        # print("primes:\n{}".format(primes))
        ps = np.hstack((ps, primes))
    end_time = time()

    print("needed time: {:.5}s".format(end_time-start_time))

    print("ps:\n{}".format(ps))

    print("ps.shape: {}".format(ps.shape))

    # cpus = 1
    cpus = mp.cpu_count()

    # def f(functions, variables, ps):
    #     variables["ps"] = ps

    # def f_calc(functions, variables, n):
    #     ps = variables["ps"]

    #     proc_num = variables["proc_num"]

    #     for i in range(0, n):
    #         a = np.dot(a, b) % 10
    #         print("proc_num: {}, i: {}".format(proc_num, i))

    #     return a

    setup_primes_o = pickle.dumps(setup_primes)
    calc_primes_o = pickle.dumps(calc_primes)

    start_time_proc_pool = time()
    proc_pool = ProcessPool(cpus)
    primes_range = np.array([2500000*i for i in range(0, cpus+1)])+1000000
    needed_primes = ps[ps<(int(np.sqrt(primes_range[-1]))+1)]
    
    proc_pool.do_new_jobs(setup_primes_o, [(needed_primes, )]*cpus)

    proc_args = [(primes_range[i], primes_range[i+1]) for i in range(0, cpus)]
    print("proc_args:\n{}".format(proc_args))
    new_primes = np.hstack(proc_pool.do_new_jobs(calc_primes_o, proc_args))
    
    proc_pool.kill_all_process()
    # b = np.vstack(proc_pool.do_new_jobs(f_calc_o, [(n, ) for _ in range(0, cpus)]))
    ps = np.hstack((ps, new_primes))
    end_time_proc_pool = time()

    print("ps.shape: {}".format(ps.shape))

    print("Taken time for proc_pool: {:.3f}s".format(end_time_proc_pool-start_time_proc_pool))

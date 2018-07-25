#! /usr/bin/python3

# -*- coding: utf-8 -*-

import dill
import marshal
import pickle
import os

import multiprocessing as mp
import numpy as np

from dotmap import DotMap
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
    # print("proc_num: {}, ps_choosen.shape: {}".format(proc_num, ps_choosen.shape))

    primes = pseudo_primes[~(np.sum(np.mod.outer(pseudo_primes, ps_choosen)==0, axis=1) > 0)]

    # print("proc_num: {}, primes: {}".format(proc_num, primes))

    return primes[primes<n_end]


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

    primes = pseudo_primes[~(np.sum(np.mod.outer(pseudo_primes, ps_choosen)==0, axis=1) > 0)]

    return primes[primes<n_end]

if __name__ == "__main__":
    from os.path import expanduser
    home = expanduser("~")
    print("home: {}".format(home))

    path_files = home+"/Documents/"
    file_path = path_files+"primes.pkl"

    # dm = DotMap()
    # dm.ps = np.array([2, 3, 5, 7])
    # dm.ps = np.hstack((dm.ps, calc_primes_standard(dm.ps, 8, 50)))
    # dm.ps = np.hstack((dm.ps, calc_primes_standard(dm.ps, 50, 1000)))
    # dm.max_num = 8

    # with open(path_files+"primes_default.pkl", "wb") as fout:
    #     dill.dump(dm, fout)

    if not os.path.exists(file_path):
        dm = DotMap()
        ps = np.array([2, 3, 5, 7])

        # numbers_range = np.array([8, 45, 1000, 10000, 100000, 1000000])
        numbers_range = np.array([8, 45]+[10**i for i in range (3, 7)])
        dm.max_num = numbers_range[-1]

        start_time = time()
        for i in range(0, len(numbers_range)-1):
            print("i: {}".format(i))
            primes = calc_primes_standard(ps, numbers_range[i], numbers_range[i+1])
            # print("primes:\n{}".format(primes))
            ps = np.hstack((ps, primes))
        end_time = time()

        print("needed time: {:.5}s".format(end_time-start_time))

        print("ps:\n{}".format(ps))

        print("ps.shape: {}".format(ps.shape))
        
        dm.ps = ps
        with open(file_path, "wb") as fout:
            dill.dump(dm, fout)

        print("Created new file: {}".format(file_path))

    with open(file_path, "rb") as fin:
        dm = dill.load(fin)

    ps = dm.ps

    cpus = mp.cpu_count()

    setup_primes_o = pickle.dumps(setup_primes)
    calc_primes_o = pickle.dumps(calc_primes)

    start_time_proc_pool = time()
    proc_pool = ProcessPool(cpus)
    # primes_range_default = np.array([20*i for i in range(0, cpus+1)])
    primes_range_default = np.array([160000*i for i in range(0, cpus+1)])

    add_offset = dm.max_num

    for i in range(0, 10):
        print("i: {}".format(i))
        primes_range = primes_range_default+add_offset
        needed_primes = ps[ps<(int(np.sqrt(primes_range[-1]))+1)]
        
        proc_pool.do_new_jobs(setup_primes_o, [(needed_primes, )]*cpus)
        proc_args = [(primes_range[i], primes_range[i+1]) for i in range(0, cpus)]
        print("proc_args:\n{}".format(proc_args))
        new_primes = np.hstack(proc_pool.do_new_jobs(calc_primes_o, proc_args))
        print("new_primes:\n{}".format(new_primes))
        add_offset = primes_range[-1]
        
        ps = np.hstack((ps, new_primes))
        print("ps.shape: {}".format(ps.shape))

    proc_pool.kill_all_running_process()
    
    end_time_proc_pool = time()

    print("ps.shape: {}".format(ps.shape))
    print("ps: {}".format(ps))
    print("Taken time for proc_pool: {:.3f}s".format(end_time_proc_pool-start_time_proc_pool))

    dm.max_num = add_offset
    dm.ps = ps
    with open(file_path, "wb") as fout:
        dill.dump(dm, fout)

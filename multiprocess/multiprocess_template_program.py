#! /usr/bin/python3

# -*- coding: utf-8 -*-

import marshal
import pickle

import multiprocessing as mp
import numpy as np

from multiprocessing import Process, Queue

from time import time

class ProcessPool(Exception):
    @staticmethod
    def process_worker(queue_in, queue_out):
        proc_num,  = queue_in.get()

        functions = {}
        variables = {"proc_num": proc_num}

        while True:
            command, f_o, args = queue_in.get()
            # command, all_args = queue_in.get()

            if command == "EXIT":
                break
            elif command == "CALC":

            #     f_name, v_names, other_args = all_args
            #     f = functions[f_name]
            #     args = []
            #     for k in v_names:
            #         args.append(variables[k])
            #     if len(other_args):
            #         args.extend(other_args)

                f = pickle.loads(f_o)
                if len(args) > 0:
                    ret = f(functions, variables, *args)
                else:
                    ret = f(functions, variables)
                queue_out.put((proc_num, ret))
            # elif command == "FUNCTIONS":
            #     for k, v in all_args:
            #         functions[k] = pickle.loads(v)
            # elif command == "VARIABLES":
            #     for k, v in all_args:
            #         variables[k] = v

    def __init__(self, process_amount):
        self.process_amount = process_amount

        self.queues_main_worker = np.array([Queue() for _ in range(0, process_amount)])
        self.queues_worker_main = np.array([Queue() for _ in range(0, process_amount)])
        self.procs = np.array([Process(target=self.process_worker, args=(queue_in, queue_out)) for
            queue_in, queue_out in zip(self.queues_main_worker, self.queues_worker_main)])

        for proc in self.procs:
            proc.start()

        for i, queue_main_worker in enumerate(self.queues_main_worker):
            queue_main_worker.put((i, ))

    def do_new_jobs(self, f_o, list_args):
        for queue_main_worker, args in zip(self.queues_main_worker, list_args):
            queue_main_worker.put(("CALC", f_o, args))

        list_ret = []
        for queue_worker_main in self.queues_worker_main:
            proc_num, ret = queue_worker_main.get()
            list_ret.append(ret)

        return list_ret

    def kill_all_running_process(self):
        for queue_main_worker in self.queues_main_worker:
            queue_main_worker.put(("EXIT", None, None))
        
        for proc in self.procs:
            proc.join()

if __name__ == "__main__":
    rows = 10000
    cols = 200
    A = np.random.randint(0, 10, (rows, cols))
    B = np.random.randint(0, 10, (cols, cols))

    cpus = mp.cpu_count()
    idx = np.arange(0, cpus+1)*(rows//cpus)
    A_split = [A[idx[i]:idx[i+1]] for i in range(0, cpus)]

    list_args = [(a, B) for a in A_split]

    def f(functions, variables, a, b):
        variables["a"] = a
        variables["b"] = b

    def f_calc(functions, variables, n):
        a = variables["a"]
        b = variables["b"]
        proc_num = variables["proc_num"]

        for i in range(0, n):
            a = np.dot(a, b) % 10
            print("proc_num: {}, i: {}".format(proc_num, i))

        return a

    f_o = pickle.dumps(f)
    f_calc_o = pickle.dumps(f_calc)

    n = 20

    start_1 = time()
    a = A
    for i in range(0, n):
        a = np.dot(a, B) % 10
        print("single_core: i: {}".format(i))
    end_1 = time()

    start_2 = time()
    proc_pool = ProcessPool(cpus)
    proc_pool.do_new_jobs(f_o, list_args)
    b = np.vstack(proc_pool.do_new_jobs(f_calc_o, [(n, ) for _ in range(0, cpus)]))
    end_2 = time()

    print("Taken time for np.dot: {:.3f}s".format(end_1-start_1))
    print("Taken time for proc_pool: {:.3f}s".format(end_2-start_2))

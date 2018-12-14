#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import numpy as np

from functools import reduce

from multiprocessing import Process, Pipe

from sympy.ntheory import factorint

import time


get_new_rnd_number = lambda arr: reduce(lambda a, b: a|b, [int(x)<<(i*8) for i, x in enumerate(arr)], 0)

# def create_new_arr(seed)

if __name__ == "__main__":
    # for i in range(0, 100):
    #     print("\ni: {}".format(i))
    #     amount_bytes = np.random.randint(1, 20)
    #     print("amount_bytes: {}".format(amount_bytes))
    #     n = get_new_rnd_number(amount_bytes)
    #     print("n: {}".format(n))

    found_primes = []

    for i in range(0, 100):
        print("i: {}".format(i))

        pipes_out, pipes_in = list(zip(*[Pipe() for _ in range(0, 3)]))

        def func_create_new_prime(pipe_in, seeds):
            primes = factorint(get_new_rnd_number(seeds))
            p = sorted(list(primes.keys()))[-1]
            pipe_in.send(p)

        ps = [Process(
                target=func_create_new_prime,
                args=(pipe_in, np.random.randint(0, 256, (35, ))))
            for pipe_in in pipes_in
        ]

        for p in ps:
            p.start()

        time.sleep(3)
        for p in ps:
            p.terminate()

        for pipe_out in pipes_out:
            if pipe_out.poll():
                prime = pipe_out.recv()
                print("A new prime was found!")
                print("prime: {}".format(prime))
                found_primes.append(prime)
            else:
                print("No Prime found!")
        
        for p in ps:
            p.join()

    print("found_primes: {}".format(found_primes))

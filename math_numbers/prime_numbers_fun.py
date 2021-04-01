#! /usr/bin/python2.7

import sys

import numpy as np

def get_primes(n):
    if n < 2:
        return
    yield 2
    if n < 3:
        return
    yield 3
    if n < 5:
        return
    yield 5
    ps = [2, 3, 5]
    d = [4, 2]

    i = 7
    j = 0
    while i <= n:
        q = int(np.sqrt(i))+1

        is_prime = True

        k = 0
        v = ps[k]
        while v < q:
            if i % v == 0:
                is_prime = False
                break
            k += 1
            v = ps[k]

        if is_prime:
            yield i
            ps.append(i)

        i += d[j]
        j = (j+1) % 2 


def prime_factorization(n, ps):
    lst = []

    for p in ps:
        count = 0
        while n%p==0:
            count += 1
            n //= p
        lst.append((p, count))
        if n==1:
            break

    return lst


def get_prime_amount_timetable(n, ps):
    timetable = np.zeros((len(ps), )).astype(np.int)

    for i, p in enumerate(ps):
        t = n//p
        j = 0
        while t > 0:
            t //= p
            j += 1
        timetable[i] = j

    return timetable


def sequence_1():
    n_max = 100
    lst = []

    for n in range(1, n_max+1):
        ps = list(get_primes(n))
        timetable = get_prime_amount_timetable(n, ps).astype(object)
        ps_pow = ps**timetable
        biggest_n_number = np.prod(ps_pow)
        lst.append(biggest_n_number)

    # sequence A003418
    print("lst: {}".format(lst))


def sequence_2():
    n_max = 100
    lst = []

    ps = list(get_primes(n_max))
    for n in range(1, n_max+1):
        prime_factors = np.array(prime_factorization(n, ps))
        print("n: {}".format(n))
        n_mult = np.sum(np.multiply.reduce(prime_factors, axis=1))
        print("n_mult: {}".format(n_mult))
        lst.append(n_mult)

    # sequence A001414
    print("lst: {}".format(lst))


def sequence_3():
    pass


if __name__ == "__main__":
    n_max = 10000000
    l_primes = list(get_primes(n_max))
    
    print("n_max: {}".format(n_max))
    print("len(l_primes): {}".format(len(l_primes)))

    sys.exit()

    # sequence_1()
    sequence_2()

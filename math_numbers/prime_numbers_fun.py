#! /usr/bin/python2.7

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
        while ps[k] < q:
            if i % ps[k] == 0:
                is_prime = False
                break
            k += 1

        if is_prime:
            yield i
            ps.append(i)

        i += d[j]
        j = (j+1) % 2 

    # return ps


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


if __name__ == "__main__":
    n = 100
    print("n: {}".format(n))

    ps = get_primes(n)
    print("ps: {}".format(ps))

    timetable = get_prime_amount_timetable(n, ps).astype(object)
    print("timetable: {}".format(timetable))

    ps_pow = ps**timetable
    print("ps_pow: {}".format(ps_pow))

    biggest_n_number = np.prod(ps_pow)
    print("biggest_n_number: {}".format(biggest_n_number))

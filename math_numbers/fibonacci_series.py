#! /usr/bin/python3.6

import decimal

from decimal import Decimal as Dec

decimal.getcontext().prec = 100

def fib_series(n):
    a = 0
    b = 1
    s = 0
    for i in range(1, n+1):
        s += 1/b

        print("i: {}, b: {}, s: {}".format(i, b, s))

        t = a+b
        a = b
        b = t

    return s

def fib_series_dec(n):
    a = 0
    b = 1
    s = Dec(0)
    for i in range(1, n+1):
        s += 1/Dec(b)

        print("i: {}, b: {}, s: {}".format(i, b, s))

        t = a+b
        a = b
        b = t

    return s

def lin_sequence_series_dec(n):
    a = 1
    s = Dec(0)
    for i in range(1, n+1):
        s += 1/Dec(a)

        print("i: {}, a: {}, s: {}".format(i, a, s))

        a = a*2+1

    return s

def lin_of_lin_sequence_series_dec(n):
    a = 1
    b = 1

    changes = 0
    choosen = 0
    increments = []
    s = Dec(0)
    dec_1 = Dec(1)
    for i in range(1, n+1):
        amount = 1
        if choosen == 0:
            a = 2*a+1
            while a < b:
                a = 2*a+1
                amount += 1
            increments.append((amount, "a"))
            s += dec_1/Dec(a)
        elif choosen == 1:
            b = 3*b+1
            while b < a:
                b = 3*b+1
                amount += 1
            increments.append((amount, "b"))
            s += dec_1/Dec(b)

        choosen = (choosen+1)%2
        changes += 1
        print("i: {}, s: {}".format(i, s))

    print("increments:\n{}".format(increments))
    globals()["increments"] = increments
    return s


if __name__ == "__main__":
    print("Hello World!")

    n = 30
    # print("get the fibonacci series with n: {}:".format(n))
    # s = fib_series_dec(n)
    
    # print("get the sequence series with n: {}:".format(n))
    # s = lin_sequence_series_dec(n)

    print("get the lin of lin sequence series with n: {}:".format(n))
    s = lin_of_lin_sequence_series_dec(n)

#! /usr/bin/python3

from fractions import Fraction as frac
from math import floor as fl

if __name__=='__main__':
    # solve x*floor(x*floor(x*floor(x))) = n, where n = 2020 e.g.

    def f(x):
        return x*fl(x*fl(x*fl(x)))

    n = 2020
    numer = 1
    denom = 1
    # a = frac(1, 1)
    is_increment_numerator = True
    while True:
        a = frac(numer, denom)
        y = f(a)
        fl_y = fl(y)
        print("numer: {}, denom: {}, float(y): {}".format(numer, denom, float(y)))
        
        if y.numerator%y.denominator==0 and fl_y==n:
            break

        if is_increment_numerator:
            numer += 1
            a_new = frac(numer, denom)
            # fl_a_new = fl(f(a_new))

            if f(a_new)>n:
            # if fl_a_new>n:
                is_increment_numerator = False
            a = a_new
        else:
            denom += 1
            a_new = frac(numer, denom+1)
            # fl_a_new = fl(f(a_new))

            if f(a_new)<n:
            # if fl_a_new<n:
                is_increment_numerator = True
            a = a_new

#! /usr/bin/python3.6

import numpy as np

def harmonic_series(n):
    s = 0
    for i in range(1, n+1):
        s += 1./i
    return s

def series_of_harmonic_series(n):
    s = 0
    for i in range(1, n+1):
        s1 = 0
        for j in range(1, i+1):
            s1 += 1./j
        s += s1
    return s

def series_of_harmonic_series_2(n):
    s = 0
    for i in range(1, n+1):
        s += 1./i*(n-i+1)
    return s

def sum_reciproc_of_shs(n):
    s = 0
    for i in range(1, n+1):
        s1 = 0
        for j in range(1, i+1):
            s1 += 1./i*(i-j+1)
        s += 1./s1
    return s

if __name__ == "__main__":
    for n in range(1, 101):
        hs = harmonic_series(n)
        shs = series_of_harmonic_series(n)
        shs2 = series_of_harmonic_series_2(n)
        srshs = sum_reciproc_of_shs(n)
        print("\nn: {}, hs: {}, shs: {}, shs2: {}".format(n, hs, shs, shs2))
        print("srshs: {}".format(srshs))

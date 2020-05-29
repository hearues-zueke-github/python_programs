#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

from copy import deepcopy

sys.path.append("../combinatorics/")
import different_combinations as combinations
sys.path.append("../math_numbers/")
from prime_numbers_fun import get_primes

from time import time

from PIL import Image

import numpy as np

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

def get_int_sqrt(n):
    # print("get_int_sqrt from: n: {}".format(n))
    n_prev_2 = n
    n_prev = n
    n_next = int((n+n/n)/2)
    # print("n_prev: {}, n_next: {}".format(n_prev, n_next))
    # input("ENTER...")
    while n_prev != n_next and n_prev_2 != n_next:
        n_prev_2 = n_prev
        n_prev = n_next
        n_next = int((n_prev+n/n_prev)/2)
        # print("n_prev: {}, n_next: {}".format(n_prev, n_next))
        # input("ENTER...")
    return n_next


def get_primes(n):
    ps = [2, 3, 5]
    p = 7

    while p <= n:
        p_sqrt = get_int_sqrt(p)+1
        is_prime = True
        for p2 in ps:
            if p % p2 == 0:
                is_prime = False
                break
            if p2 > p_sqrt:
                break
        if is_prime:
            ps.append(p)

        p += 2

    return ps


def determinante(A):
    if A.shape == (1, 1):
        return A[0, 0]
    elif A.shape == (2, 2):
        return A[0, 0]*A[1, 1]-A[1, 0]*A[0, 1]
    elif A.shape == (3, 3):
        return (A[0, 0]*A[1, 1]*A[2, 2]+A[0, 1]*A[1, 2]*A[2, 0]+A[0, 2]*A[1, 0]*A[2, 1]
               -A[2, 0]*A[1, 1]*A[0, 2]-A[2, 1]*A[1, 2]*A[0, 0]-A[2, 2]*A[1, 0]*A[0, 1])
    elif len(A.shape) == 2 and A.shape[0] == A.shape[1]:
        result = 0
        n = A.shape[0]
        l = list(range(0, n))
        for i in range(0, n):
            l.pop(i)
            B = A[l, 1:]
            l.insert(i, i)
            if i % 2 == 0:
                result += A[i, 0]*determinante(B)
            else:
                result -= A[i, 0]*determinante(B)
        return result
    return None


def solve_system_of_linear_equations(A, x, B):
    n = A.shape[0]
    if n < 0:
        raise Exception("A must have at least one row!")
    if n != A.shape[1]:
        raise Exception("A must be a square matrix!")
    if n != x.shape[0]:
        raise Exception("x does not have the same row size like A!")
    if n != B.shape[0]:
        raise Exception("B does not have the same row size like A!")

    A = A.copy()
    A_orig = A.copy()

    det_A = determinante(A)
    if det_A == 0:
        return 1

    for i in range(0, n):
        A[:, i] = B
        det_Ax = determinante(A)
        val = det_Ax % det_A
        if val != 0:
            return 2
        x[i] = det_Ax // det_A
        A[:, i] = A_orig[:, i]

    return 0

class Polynomials(Exception):
    def __init__(self, factors=[]):
        factors = list(factors)

        is_found_zeros_right = False
        for i in range(len(factors)-1, -1, -1):
            if factors[i] != 0:
                break
            is_found_zeros_right = True

        self.factors = factors


    def __len__(self):
        return len(self.factors)


    def __add__(self, other):
        factors1 = list(self.factors)
        factors2 = list(other.factors)

        l1 = len(factors1)
        l2 = len(factors2)

        if l1 > l2:
            factors2.extend([0]*(l1-l2))
        elif l1 < l2:
            factors1.extend([0]*(l2-l1))

        new_factors_arr = np.array(factors1)+np.array(factors2)
        return Polynomials(factors=new_factors_arr.tolist())


    def __sub__(self, other):
        factors1 = list(self.factors)
        factors2 = list(other.factors)

        l1 = len(factors1)
        l2 = len(factors2)

        if l1 > l2:
            factors2.extend([0]*(l1-l2))
        elif l1 < l2:
            factors1.extend([0]*(l2-l1))

        new_factors_arr = np.array(factors1)-np.array(factors2)
        return Polynomials(factors=new_factors_arr.tolist())


    def __truediv__(self, other):
        return self+other


    def __mod__(self, other):
        return self+other


    def shift_factors(self, n):
        if n == 0:
            return
        elif n < 0:
            n = -n
            if n > len(self.factors):
                self.factors = []
            else:
                self.factors = self.factors[n:]
            return

        self.factors = [0]*n+self.factors


    def findFactors(self):
        A = np.zeros((3, 3), dtype=np.int)
        A[0, 0] = 1
        A[1] = 1**np.arange(0, 3)
        A[2] = 2**np.arange(0, 3)

        print("A:\n{}".format(A))

        f1 = self.f(0)
        f2 = self.f(1)
        f3 = self.f(2)

        print("f1: {}".format(f1))
        print("f2: {}".format(f2))
        print("f3: {}".format(f3))

        fs = np.abs(np.array([f1, f2, f3]))
        max_f = np.max(fs)
        print("max_f: {}".format(max_f))

        ps = get_primes(get_int_sqrt(max_f)+1)
        print("ps: {}".format(ps))

        def get_prime_factors(n, ps):
            if n == 0:
                return [0]
            if n == 1:
                return [1]

            factors = []
            for p in ps:
                while n % p == 0:
                    factors.append(p)
                    n //= p
            if n > 1:
                factors.append(n)
            return factors

        p_factors1 = get_prime_factors(np.abs(f1), ps)
        p_factors2 = get_prime_factors(np.abs(f2), ps)
        p_factors3 = get_prime_factors(np.abs(f3), ps)

        print("p_factors1: {}".format(p_factors1))
        print("p_factors2: {}".format(p_factors2))
        print("p_factors3: {}".format(p_factors3))

        lst_factors = [p_factors1, p_factors2, p_factors3]

        def get_all_combos(p_factors):
            if len(p_factors) == 1:
                if p_factors[0] == 0:
                    return [0]
                elif p_factors[0] == 1:
                    return [1, -1]
                else:
                    v = p_factors[0]
                    return [v, -v]

            factors = []
            lens = []

            for p in p_factors:
                if p in factors:
                    lens[factors.index(p)] += 1
                else:
                    factors.append(p)
                    lens.append(1)
            print("factors: {}, lens: {}".format(factors, lens))

            l = len(factors)
            values = []
            positions = [0 for _ in range(0, l)]

            positions[0] = -1

            is_finished_cycle = False
            while not is_finished_cycle:
                i = 0
                need_more_cycles = True
                while i < l and need_more_cycles:
                    positions[i] += 1
                    if positions[i] > lens[i]:
                        positions[i] = 0
                        i += 1
                    else:
                        need_more_cycles = False

                if i >= l:
                    is_finished_cycle = True
                else:
                    s = 1
                    for j in range(0, l):
                        s *= factors[j]**positions[j]
                    values.append(s)

            negatives = [-v for v in values]
            return values+negatives

        lsts_factors = []
        for p_factors in lst_factors:
            print("p_factors: {}".format(p_factors))
            lst_combos = get_all_combos(p_factors)
            print("lst_combos: {}".format(lst_combos))
            lsts_factors.append(lst_combos)

        print("lsts_factors:\n{}".format(lsts_factors))

        l = len(lsts_factors)
        lens = [len(lst_factors) for lst_factors in lsts_factors]
        positions = [0 for _ in range(0, l)]
        positions[0] = -1

        x = np.zeros((3, ), dtype=np.int)

        is_finished_cycle = False
        while not is_finished_cycle:
            i = 0
            need_more_cycles = True
            while i < l and need_more_cycles:
                positions[i] += 1
                if positions[i] >= lens[i]:
                    positions[i] = 0
                    i += 1
                else:
                    need_more_cycles = False

            if i >= l:
                is_finished_cycle = True
            else:
                values = [lst_factors[positions[i]] for i, lst_factors in enumerate(lsts_factors, 0)]
                B = np.array(values)

                ret = solve_system_of_linear_equations(A, x, B)
                if ret == 0:
                    print("B: {}".format(B))
                    print("x: {}".format(x))
                    print("A.dot(x): {}".format(A.dot(x)))
                    
                    if 0 in x:
                        is_finished_cycle = True
                        print("FOUND ONE WITH 0 in x!")

                print("positions: {}, values: {}".format(positions, values))


    def f(self, x):
        s = 0
        for i, a in enumerate(self.factors, 0):
            s += a*x**i
        return s


    def __str__(self):
        lst = []
        factors = self.factors
        for i in range(0, len(factors)):
            f = factors[i]
            if f != 0:
                if f > 0:
                    fs = "{}".format(f)
                else:
                    fs = "({})".format(f)

                if i >= 2:
                    lst.append("{}*x^{}".format(fs, i))
                elif i == 1:
                    lst.append("{}*x".format(fs))
                else:
                    lst.append(fs)

        if len(lst) == 0:
            return "0"

        return " + ".join(lst)


    def do_div_mod(self, other):
        pl = Polynomials(factors=self.factors)
        pr = Polynomials(factors=other.factors)

        l1 = len(pl.factors)
        l2 = len(pr.factors)


        if l1 < l2:
            q = Polynomials(factors=[])
            r = pl
        else:
            

        return q, r


if __name__ == "__main__":
    print("Hello World!")

    p1 = Polynomials(factors=[2, 3, 4, -1])
    p2 = Polynomials(factors=[-3, 4, 4, -1, 12])
    print("p1: {}".format(p1))
    print("p2: {}".format(p2))

    print("p1+p2: {}".format(p1+p2))
    print("p1-p2: {}".format(p1-p2))
    print("p2+p1: {}".format(p2+p1))
    print("p2-p1: {}".format(p2-p1))
    
    print("\np1/p2: {}".format(p1/p2))
    print("p1%p2: {}".format(p1%p2))

    p = Polynomials(factors=np.random.randint(-10, 11, (5, )))
    p.findFactors()

    # n = 3
    # ret = 1
    # while ret != 0:
    #     A = np.random.randint(-10, 11, (n, n))

    #     x = np.zeros((n, ), dtype=np.int)
    #     B = np.random.randint(-10, 11, (n, ))

    #     # print("A:\n{}".format(A))
    #     # print("np.linalg.det(A): {}".format(np.linalg.det(A)))
    #     # print("determinante(A): {}".format(determinante(A)))
    #     # print("B: {}".format(B))
        
    #     ret = solve_system_of_linear_equations(A, x, B)
    #     if ret != 0:
    #         print("Not a x found!")
    #     else:
    #         print("A:\n{}".format(A))
    #         print("np.linalg.det(A): {}".format(np.linalg.det(A)))
    #         print("determinante(A): {}".format(determinante(A)))
    #         print("B: {}".format(B))
    #         print("x: {}".format(x))

    #         assert(np.all(A.dot(x)==B))

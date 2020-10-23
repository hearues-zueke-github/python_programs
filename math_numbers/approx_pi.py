#! /usr/bin/python3

# -*- coding: utf-8 -*-

import decimal
from decimal import Decimal
decimal.getcontext().prec = 80

import numpy as np

import matplotlib.pyplot as plt
plt.close("all")

PI = Decimal('3.14159265358979323846264338327950288419716939937510')
EPSILON = Decimal('1E-12')

def get_approx_pi_method_1(n):
    dec_2 = Decimal(2)
    ys = [dec_2]

    y = dec_2
    z = np.sqrt(dec_2)
    y *= dec_2/z
    ys.append(y)

    for j in range(2, n+1):
        z = np.sqrt(dec_2+z)
        y *= dec_2/z
        ys.append(y)

    return ys


def get_approx_pi_method_2(n):
    y1 = Decimal(0)
    y2 = Decimal(0)
    ys = []

    for k in range(0, n+1):
        y1 += (-1)**k*(Decimal(1)/5)**(2*k+1)/(2*k+1)
        y2 += (-1)**k*(Decimal(1)/239)**(2*k+1)/(2*k+1)
        y = 16*y1-4*y2
        ys.append(y)

    return ys


def get_approx_pi_method_3(n):
    y = Decimal(0)
    ys = []

    for k in range(0, n+1):
        y += 1/Decimal(16**k)*(Decimal(4)/(8*k+1)-Decimal(2)/(8*k+4)-Decimal(1)/(8*k+5)-Decimal(1)/(8*k+6))
        ys.append(y)

    return ys


def get_approx_pi_method_own_1(n):
    y1 = Decimal(1)
    y2 = Decimal(1)
    ys = [y1/y2*2]

    for k in range(1, n+1):
        y1 += 1/Decimal(2*k-1)
        y2 += 1/Decimal(2*k)
        # y += 1/Decimal(16**k)*(Decimal(4)/(8*k+1)-Decimal(2)/(8*k+4)-Decimal(1)/(8*k+5)-Decimal(1)/(8*k+6))
        y = y1/y2*2
        # print("k: {}, y: {}".format(k, y))
        ys.append(y)

    return ys


def get_needed_n_with_epsilon_precision(ys, epsilon):
    for i, y in enumerate(ys, 0):
        if np.abs(y-PI)<epsilon:
            break
    return i


if __name__ == '__main__':
    print("Real pi: {}".format(PI))

    n = 30

    ys1 = np.array(get_approx_pi_method_1(n))
    ys2 = np.array(get_approx_pi_method_2(n))
    ys3 = np.array(get_approx_pi_method_3(n))
    ys_own_1 = np.array(get_approx_pi_method_own_1(100))

    print("ys1[-1]: {}".format(ys1[-1]))
    print("ys2[-1]: {}".format(ys2[-1]))
    print("ys3[-1]: {}".format(ys3[-1]))
    print("ys_own_1[-1]: {}".format(ys_own_1[-1]))


    # find needed n for getting EPSILON precision!
    n1 = get_needed_n_with_epsilon_precision(ys1, EPSILON)
    n2 = get_needed_n_with_epsilon_precision(ys2, EPSILON)
    n3 = get_needed_n_with_epsilon_precision(ys3, EPSILON)

    print("Needed terms for approx method nr. 1: {}".format(n1))
    print("Needed terms for approx method nr. 2: {}".format(n2))
    print("Needed terms for approx method nr. 3: {}".format(n3))

    
    # create the plot of the absolute differences with the real number pi
    plt.figure()

    plt.yscale('log')
    plt.title('Absolute differences of\ndifferent approximations for pi')

    xs1 = np.arange(0, len(ys1))
    xs2 = np.arange(0, len(ys2))
    xs3 = np.arange(0, len(ys3))

    diff_ys1 = np.abs(ys1-PI)
    diff_ys2 = np.abs(ys2-PI)
    diff_ys3 = np.abs(ys3-PI)

    p1 = plt.plot(xs1, diff_ys1, 'b.-')[0]
    p2 = plt.plot(xs2, diff_ys2, 'g.-')[0]
    p3 = plt.plot(xs3, diff_ys3, 'r.-')[0]

    plt.plot((0, n), (EPSILON, EPSILON), 'k-', linewidth=0.5)

    plt.legend((p1, p2, p3), ('Approx method nr. 1', 'Approx method nr. 2', 'Approx method nr. 3'))

    plt.xlabel('n')
    plt.ylabel('absolute difference')

    plt.xlim([0, n])

    # plt.show()
    plt.savefig('absolute_differences.png')

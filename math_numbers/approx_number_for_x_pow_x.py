#! /usr/bin/python3.5

# approx the value of x**x = y, for x

import numpy as np

import matplotlib.pyplot as plt

def calc_first_derivative(f, x):
    epsilon = 0.00001
    return (f(x+epsilon)-f(x-epsilon))/(2*epsilon)

def calc_second_derivative(f, x):
    epsilon = 0.00001
    fp = calc_first_derivative(f, x+epsilon)
    fm = calc_first_derivative(f, x-epsilon)
    return (fp-fm)/(2*epsilon)

y = 1000.

f = lambda x: (y-x**x)**2

# xs = [i for i in np.arange(0.1, 4, 0.1)]
# ys = [x**x for x in xs]
# ysd = [f(x) for x in xs]
# ysd1 = [calc_first_derivative(f, x) for x in xs]
# ysd2 = [calc_second_derivative(f, x) for x in xs]

# plt.figure()

# plt.xlabel("x")
# plt.ylabel("f(x)")

# plt.plot(xs, ys, "-")
# plt.plot(xs, ysd, "-")
# plt.plot(xs, ysd1, "-")
# plt.plot(xs, ysd2, "-")

# plt.ylim([-0.5, 1000])

# ax = plt.gca()
# # ax.set_yscale("log", nonposy='clip')

# plt.show()

x = 1.

for i in range(0, 28000):
    fd1 = calc_first_derivative(f, x)
    # fd2 = calc_second_derivative(f, x)
    # x = x - 1.2* fd1 / fd2
    x = x - 0.00000001* fd1
    xx = x**x

    # if xx > y:
    #     x = x - 0.0001*(y-xx)**2
    # else:
    #     x = x + 0.0001*(y-xx)**2

    # print("x: {}, xx: {}, fd1: {}, fd2: {}".format(x, xx, fd1, fd2))
    print("i: {}, x: {}, xx: {}, fd1: {}".format(i, x, xx, fd1))

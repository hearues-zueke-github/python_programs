#! /usr/bin/python2.7

import numpy as np

import matplotlib.pyplot as plt

ls1 = [3.8, 1.5]
ls2 = [3.7, 1.1]

x1 = 0.5
x2 = 0.5

f = lambda l1, l2, x: l1*x*(1-x*l2)

def get_points(lamb1, lamb2, x1, x2, iterations):
    xs = np.zeros((iterations, ))
    ys = np.zeros((iterations, ))

    for i in xrange(0, iterations):
        x1 = f(ls1[0], ls1[1], x1)
        xs[i] = x1
        x2 = f(ls2[0], ls2[1], x2)
        ys[i] = x2

    return xs, ys

iterations = 30000
i = 0
add = 0
    
plt.figure()
for add in np.arange(0.35, 2., 0.02):
    plt.clf()
    print("i: {}".format(i))
    # ls2_0 = ls2[0]+add
    ls1[1] = np.round(add, 2)
    xs, ys = get_points(ls1, ls2, x1, x2, iterations)
    plt.title("ls1: {}, ls2: {}, iters: {}".format(ls1, ls2, iterations))
    plt.plot(xs, ys, "b.", markersize=0.5)
    del xs
    del ys
    plt.xlim((0., 3.))
    # plt.ylim((0, 1.))
    plt.savefig("random_dots_pictures/i_{:02d}_ls1_{}_{}_ls2_{}_{}.png".format(i, ls1[0], ls1[1], ls2[0], ls2[1]), format="png", dpi=400)
    i += 1

# plt.show()

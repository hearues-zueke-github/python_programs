#! /usr/bin/python2.7

import numpy as np

import matplotlib.pyplot as plt

lamb1 = 3.95
lamb2 = 3.5

start_x_1 = 0.5
start_x_2 = 0.5
x1 = start_x_1
x2 = start_x_2

iterations = 3000
f = lambda lamb, x: lamb*x*(1-x)

def get_points(lamb1, lamb2, x1, x2, iterations):
    xs = np.zeros((iterations, ))
    ys = np.zeros((iterations, ))

    for i in xrange(0, iterations):
        x1 = f(lamb1, x1)
        xs[i] = x1
        x1 = f(lamb2, x1)
        # x2 = f(lamb2, x2)
        # xs[i] = x1
        ys[i] = x1

    return xs, ys

i = 0
    
plt.figure()
for add in np.arange(0., 0.5, 0.01):
    print("i: {}".format(i))
    lamb2new = lamb2+add
    xs, ys = get_points(lamb1, lamb2new, x1, x2, iterations)
    plt.title("lambda1: {}, lambda2: {}, iters: {}".format(lamb1, lamb2new, iterations))
    plt.plot(xs, ys, "b.", lw=0.01)
    del xs
    del ys
    plt.xlim((0, 1.))
    plt.ylim((0, 1.))
    plt.savefig("feigenbaum_pictures/i_{:02d}_lamb1_{}_lamb2_{}.png".format(i, lamb1, lamb2new), format="png")
    i += 1

# plt.show()

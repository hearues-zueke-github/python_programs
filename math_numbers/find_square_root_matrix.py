#! /usr/bin/python3

# # TODO: Delete afterward
# import matplotlib
# # matplotlib.use('Agg')

import pickle as pkl    # to load dataset
import numpy as np      # for math stuff
import tensorflow as tf # for graphs
import pylab as plt     # for plotins

import sys

# plt.close('all')   # closes all previous figures

dim = 3

get_error = lambda B, A: np.sum((B*B-A)**2)

while True:
    A = np.random.uniform(0., 10., (dim, dim))

    def get_B_approx(B):
        # print("before updating A =\n{}".format(A))
        # print("before updating B =\n{}".format(B))
        # print("error = {}".format(np.sum((B*B-A)**2)))
        errors = [get_error(B, A)]
        alpha = 0.0000001
        alpha_min = 0.000000005
        for i in range(0, 3000):
            nabla = (B.T.dot(B)+B.dot(B.T)).T.dot(B.dot(B)-A)
            tries = 0
            while True:
                B_new = B - alpha*nabla
                error = get_error(B, A)
                if errors[-1] >= error:
                    alpha *= 1.005
                    break
                else:
                    alpha /= 1.05
                    if alpha < alpha_min:
                        alpha = alpha_min

                    tries += 1
                    if tries > 10:
                        break
            # if tries > 10:
            #     break
            B = B_new
            errors.append(error)
            # print("i: {}, alpha: {:.6f}, error: {:7.5f}".format(i, alpha, error))
        return B, errors[-1]

    iters = 300
    Bs = np.zeros((iters, dim, dim))
    errors = np.zeros((iters, ))
    for i in range(0, iters):
        print("i: {}, ".format(i), end="")
        if (i+1) % 16 == 0:
            print("")
        # sys.stdout.write("i: {}, ".format(i))
        B = np.random.uniform(0., 10., (dim, dim))
        Bs[i], errors[i] = get_B_approx(B)

        # fig = plt.figure()
        # plt.plot(np.arange(0, len(errors)), errors, "-b")
        # plt.title("MSE of B.dot(B) problem")
        # plt.xlabel("iterations {}.format(i+1)")
        # plt.ylabel("MSE error")
    print("")

    best_index = int(sorted(np.vstack((np.arange(0, iters).reshape((1, iters)), errors.reshape((1, iters)))).T.tolist(), key=lambda x: x[1])[0][0])
    print("best_index: {}".format(best_index))
    print("best error: {}".format(errors[best_index]))

    B_best = Bs[best_index]
    error_best = get_error(B_best, A)

    print("A:\n{}".format(A))
    print("np.linalg.det(A): {}".format(np.linalg.det(A)))
    print("B_best:\n{}".format(B_best))
    print("B_best.dot(B_best):\n{}".format(B_best.dot(B_best)))
    print("error_best:\n{}".format(error_best))

    # print("after updating A =\n{}".format(A))
    # print("after updating B =\n{}".format(B))
    # print("after updating B.dot(B) =\n{}".format(B.dot(B)))

    # plt.show()

    inp_user = input("What to do next? ")

    if inp_user == "cont" or inp_user == "":
        continue
    elif inp_user == "exit":
        print("exit this program!")
        break

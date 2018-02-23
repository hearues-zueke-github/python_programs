#! /usr/bin/python2.7

import numpy as np

import matplotlib.pyplot as plt

from colorama import Fore, Style
from copy import deepcopy

def simple_grad_calculating():
    f = lambda X, A: X.dot(A)
    fd = lambda X, A, B: X.T.dot(f(X, A)-B)
    err_mse = lambda X, A, B: np.sum((f(X, A)-B)**2)/2

    def get_numerical_gradient(X, A, B):
        y, x = A.shape
        grad = np.zeros((y, x))
        epsilon = 0.0001

        for j in xrange(0, y):
            for i in xrange(0, x):
                A[j, i] += epsilon
                f1 = err_mse(X, A, B)
                A[j, i] -= epsilon*2
                f2 = err_mse(X, A, B)
                A[j, i] += epsilon

                grad[j, i] = (f1-f2)/2/epsilon

        return grad

    get_random_matrix = lambda shape: np.random.random(shape)*2-1

    n = 3
    m = 5

    A = get_random_matrix((n, n))
    X = get_random_matrix((m, n))
    B = get_random_matrix((m, n))

    print("A:\n{}".format(A))
    print("X:\n{}".format(X))
    print("B:\n{}".format(B))

    Y = f(X, A)
    print("Y:\n{}".format(Y))

    d = Y-B
    print("d:\n{}".format(d))

    grad_num = get_numerical_gradient(X, A, B)
    grad_real = fd(X, A, B)

    print("grad_num:\n{}".format(grad_num))
    print("grad_real:\n{}".format(grad_real))

def simple_gradient_descent():
    f = lambda X, A: X.dot(A)
    fd = lambda X, A, B: X.T.dot(f(X, A)-B)
    err_mse = lambda X, A, B: np.sum((f(X, A)-B)**2)/2

    def get_numerical_gradient(X, A, B):
        y, x = A.shape
        grad = np.zeros((y, x))
        epsilon = 0.0001

        for j in xrange(0, y):
            for i in xrange(0, x):
                A[j, i] += epsilon
                f1 = err_mse(X, A, B)
                A[j, i] -= epsilon*2
                f2 = err_mse(X, A, B)
                A[j, i] += epsilon

                grad[j, i] = (f1-f2)/2/epsilon

        return grad

    get_random_matrix = lambda shape: np.random.random(shape)*2-1

    m = 100
    k = 6
    n = 20

    A1 = get_random_matrix((k, n))
    X = get_random_matrix((m, k))
    B = X.dot(A1)+get_random_matrix((m, n))*0.1
    A = get_random_matrix((k, n))

    Y = f(X, A)

    d = Y-B

    errors = []
    eta = 0.001
    for i in xrange(0, 10000):
        Ad = fd(X, A, B)
        Anew = A-eta*Ad
        error = err_mse(X, Anew, B)
        print("i: {}, error: {}".format(i, error))
        A = Anew
        errors.append(error)

    plt.figure()
    plt.plot(errors)
    plt.show()

def simple_gradient_descent_1_hidden_layer():
    f_sig = lambda X: 1/(1+np.exp(-X))
    fd_sig = lambda X: f_sig(X)*(1-f_sig(X))
    f_tanh = lambda X: np.tanh(X)
    fd_tanh = lambda X: 1 - np.tanh(X)**2
    
    def calc_forward(X, bws):
        ones = np.ones((X.shape[0], 1))
        Y = X
        for bw in bws[:-1]:
            Y = f_tanh(np.hstack((ones, Y)).dot(bw))
        Y = f_sig(np.hstack((ones, Y)).dot(bws[-1]))

        return Y

    def calc_backprop(X, bws, T):
        Xs = []
        Ys = [X]
        ones = np.ones((X.shape[0], 1))
        Y = X
        for i, bw in zip(xrange(len(bws), 0, -1), bws[:-1]):
            A = np.hstack((ones, Y)).dot(bw); Xs.append(A)
            Y = f_tanh(A); Ys.append(Y)
        A = np.hstack((ones, Y)).dot(bws[-1]); Xs.append(A)
        Y = f_sig(A); Ys.append(Y)

        d = fd_sig(Xs[-1])*(Y-T)
        bwsd = deepcopy(bws)
        bwsd[-1] = np.hstack((ones, Ys[-2])).T.dot(d)
        
        for i in xrange(2, len(bws)+1):
            d = d.dot(bws[-i+1][1:].T)*fd_tanh(Xs[-i])
            bwsd[-i] = np.hstack((ones, Ys[-i-1])).T.dot(d)

        return bwsd

    def calc_backprop_num(X, bws, T):
        bwds = deepcopy(bws)
        epsilon = 0.0001

        for bw, bwd in zip(bws, bwds):
            y, x = bwd.shape
            for j in xrange(0, y):
                for i in xrange(0, x):
                    bw[j, i] += epsilon
                    f1 = err_mse(X, bws, T)
                    bw[j, i] -= epsilon*2
                    f2 = err_mse(X, bws, T)
                    bw[j, i] += epsilon

                    bwd[j, i] = (f1-f2)/2/epsilon

        return bwds
    
    def get_random_bws(nl):
        return np.array([np.random.uniform(-1./np.sqrt(n)/100, 1./np.sqrt(n)/100, (m+1, n)) for m, n in zip(nl[:-1], nl[1:])])

    err_mse = lambda X, bws, T: np.sum((calc_forward(X, bws)-T)**2)/2#/X.shape[0]

    get_random_matrix = lambda shape: np.random.random(shape)*2-1

    m = 5
    k = 3
    n = 4

    nl = [k, 6, n]

    X = get_random_matrix((m, k))
    bws1 = get_random_bws(nl)*100
    T = calc_forward(X, bws1)
    bws = get_random_bws(nl)

    bwsd_num = calc_backprop_num(X, bws, T)
    bwsd_real = calc_backprop(X, bws, T)

    print("bwsd_num:\n{}".format(bwsd_num))
    print("bwsd_real:\n{}".format(bwsd_real))

    bwsd_diff = bwsd_num-bwsd_real
    print("bwsd_diff: {}".format(bwsd_diff))
    sum_test = np.sum([np.sum(bwsd_diff_i >= 10**-4) for bwsd_diff_i in bwsd_diff])
    print("sum_test: {}".format(sum_test))

    if sum_test == 0:
        print("Grad check is OK!")
    else:
        print("Something wrong in grad check!!! NOT OK!")

    raw_input("ENTER!...")

    errors = []
    eta = 0.001
    for i in xrange(0, 10000):
        bwsd = calc_backprop(X, bws, T)
        bws_new = bws-bwsd*eta
        error = err_mse(X, bws_new, T)
        print("i: {}, error: {}".format(i, error))
        bws = bws_new
        errors.append(error)

    print("bws1:\n{}".format(bws1))
    print("bws:\n{}".format(bws))

    plt.figure()
    plt.plot(errors)
    plt.show()

def simple_gradient_descent_1_hidden_layer_eta_checker():
    f_sig = lambda X: 1/(1+np.exp(-X))
    fd_sig = lambda X: f_sig(X)*(1-f_sig(X))
    f_tanh = lambda X: np.tanh(X)
    fd_tanh = lambda X: 1 - np.tanh(X)**2
    
    def calc_forward(X, bws):
        ones = np.ones((X.shape[0], 1))
        Y = X
        for bw in bws[:-1]:
            Y = f_tanh(np.hstack((ones, Y)).dot(bw))
        Y = f_sig(np.hstack((ones, Y)).dot(bws[-1]))

        return Y

    def calc_backprop(X, bws, T):
        Xs = []
        Ys = [X]
        ones = np.ones((X.shape[0], 1))
        Y = X
        for i, bw in zip(xrange(len(bws), 0, -1), bws[:-1]):
            A = np.hstack((ones, Y)).dot(bw); Xs.append(A)
            Y = f_tanh(A); Ys.append(Y)
        A = np.hstack((ones, Y)).dot(bws[-1]); Xs.append(A)
        Y = f_sig(A); Ys.append(Y)

        d = fd_sig(Xs[-1])*(Y-T)
        bwsd = deepcopy(bws)
        bwsd[-1] = np.hstack((ones, Ys[-2])).T.dot(d)
        
        for i in xrange(2, len(bws)+1):
            d = d.dot(bws[-i+1][1:].T)*fd_tanh(Xs[-i])
            d_abs = np.abs(d)
            max_num = np.max(d_abs)
            min_num = np.min(d_abs)
            if max_num < 1 and max_num > 0.000001:
                d /= max_num
            if min_num < 0.0001:
                d *= np.sqrt(d_abs)

            bwsd[-i] = np.hstack((ones, Ys[-i-1])).T.dot(d)

        return bwsd
    
    def get_random_bws(nl):
        l = [np.random.uniform(-1./np.sqrt(n), 1./np.sqrt(n), (m+1, n)) for m, n in zip(nl[:-1], nl[1:])]
        larray = np.empty(len(l), dtype=object)
        larray[:] = l
        return larray

    err_mse = lambda X, bws, T: np.sum((calc_forward(X, bws)-T)**2)/2#/X.shape[0]

    get_random_matrix = lambda shape: np.random.random(shape)*2-1

    m = 100
    k = 12
    n = 6

    nl = [k, 30, 20, 10, n]

    X = get_random_matrix((m, k))
    bws1 = get_random_bws(nl)
    T = calc_forward(X, bws1)
    bws = get_random_bws(nl)

    # check different etas!
    # eta_multipliers = 1.1**np.arange(-15, 4)
    eta_mult1 = 0.7**np.arange(6, -1, -1)
    eta_mult2 = 1.1**np.arange(1, 6)
    eta_multipliers = np.array(eta_mult1.tolist()+eta_mult1.tolist())
    eta_multipliers += (np.random.random(eta_multipliers.shape)-0.5)*0.02

    eta_max = 10.
    eta_min = 0.00001

    bws_prev = bws
    bwsd_prev = calc_backprop(X, bws, T)
    bws = bws-0.1*bwsd_prev

    def get_trained_network(bws, bws_prev, iterations, alpha):
        eta_now = 0.1
        eta_changed_min = eta_now
        eta_changed_max = eta_now
        errors = []
        etas = []
        error_prev = err_mse(X, bws, T)
        for i in xrange(0, iterations):
            bws_1 = bws+alpha*(bws-bws_prev)
            bwsd = calc_backprop(X, bws_1, T)
            # etas = np.arange(0.001, 4/np.sqrt(i+1), 0.2/np.sqrt(i+1)) # also works

            etas_now = eta_now*eta_multipliers
            # error_first = err_mse(X, bws, T)
            errors_now = np.array([err_mse(X, bws-eta*bwsd, T) for eta in etas_now])

            min_idx = np.argmin(errors_now)
            # print("min_idx: {}".format(min_idx))
            eta_now = etas_now[min_idx]
            error = errors_now[min_idx]
            

            if eta_now < eta_min:
                eta_now = eta_min
            if eta_now > eta_max:
                eta_now = eta_max

            if eta_changed_min > eta_now:
                eta_changed_min = eta_now
            if eta_changed_max < eta_now:
                eta_changed_max = eta_now

            bws = bws-eta_now*bwsd
            print("i: {}, eta_now: {:>10.7f}, , error: {:>10.7f}".format(i, eta_now, error))
            etas.append(eta_now)
            errors.append(error)
            
            # if error_prev > error and error_prev-error < 10**-6:
            #     break
            error_prev = error

            bwsd_prev = bwsd
            bws_prev = bws

        print("eta_changed_min: {}".format(eta_changed_min))
        print("eta_changed_max: {}".format(eta_changed_max))

        return bws, bws_prev, errors, etas

    # bws_1, bws_1_prev, errors, etas = get_trained_network(bws, bws_prev, 1000, 0.1)
    
    alphas = np.arange(0, 8)*0.25

    def get_plots(training_data, title):
        arr_bws_1, arr_bws_1_prev, arr_errors, arr_etas = list(zip(*training_data))

        fig, arr = plt.subplots(2, len(alphas), figsize=(20, 9))
        plt.suptitle(title, fontsize=16)
        arr[0, 0].set_title("Best Errors")
        arr[1, 0].set_title("Best Etas")
        
        error_min = np.min(arr_errors)
        error_max = np.max(arr_errors)

        eta_min = np.min(arr_etas)
        eta_max = np.max(arr_etas)

        alphas_str = ["{:4.2f}".format(alpha) for alpha in alphas]

        for i in xrange(0, len(alphas)):
            arr[0, i].set_ylim(error_min, error_max)
            arr[0, i].set_yscale("log", nonposy='clip')

            p = arr[0, i].plot(arr_errors[i], "b-")[0]
            p.set_label("alpha "+alphas_str[i])
            arr[0, i].legend()
        
            arr[1, i].set_ylim(eta_min, eta_max)
            arr[1, i].set_yscale("log", nonposy='clip')

            p = arr[1, i].plot(arr_etas[i], "b.")[0]
            p.set_label("alpha "+alphas_str[i])
            arr[1, i].legend()

        plt.subplots_adjust(left=0.02, bottom=0.05, right=0.98, top=0.92, wspace=0.1, hspace=0.15)
    
    iterations = 1500
    training_data = [get_trained_network(bws, bws_prev, iterations, alpha) for alpha in alphas]
    get_plots(training_data, "With Alpha")

    plt.show()

def advanced_gradient_descent_1_hidden_layer():
    f_sig = lambda X: 1/(1+np.exp(-X))
    fd_sig = lambda X: f_sig(X)*(1-f_sig(X))
    f_tanh = lambda X: np.tanh(X)
    fd_tanh = lambda X: 1 - np.tanh(X)**2
    
    def calc_forward(X, bws):
        ones = np.ones((X.shape[0], 1))
        Y = X
        for bw in bws[:-1]:
            Y = f_tanh(np.hstack((ones, Y)).dot(bw))
        Y = f_sig(np.hstack((ones, Y)).dot(bws[-1]))

        return Y

    def calc_backprop(X, bws, T):
        Xs = []
        Ys = [X]
        ones = np.ones((X.shape[0], 1))
        Y = X
        for i, bw in zip(xrange(len(bws), 0, -1), bws[:-1]):
            A = np.hstack((ones, Y)).dot(bw); Xs.append(A)
            Y = f_tanh(A); Ys.append(Y)
        A = np.hstack((ones, Y)).dot(bws[-1]); Xs.append(A)
        Y = f_sig(A); Ys.append(Y)

        d = fd_sig(Xs[-1])*(Y-T)
        bwsd = deepcopy(bws)
        bwsd[-1] = np.hstack((ones, Ys[-2])).T.dot(d)
        
        for i in xrange(2, len(bws)+1):
            d = d.dot(bws[-i+1][1:].T)*fd_tanh(Xs[-i])
            bwsd[-i] = np.hstack((ones, Ys[-i-1])).T.dot(d)

        return bwsd

    def calc_backprop_num(X, bws, T):
        bwds = deepcopy(bws)
        epsilon = 0.0001

        for bw, bwd in zip(bws, bwds):
            y, x = bwd.shape
            for j in xrange(0, y):
                for i in xrange(0, x):
                    bw[j, i] += epsilon
                    f1 = err_mse(X, bws, T)
                    bw[j, i] -= epsilon*2
                    f2 = err_mse(X, bws, T)
                    bw[j, i] += epsilon

                    bwd[j, i] = (f1-f2)/2/epsilon

        return bwds
    
    def get_random_bws(nl):
        return np.array([np.random.uniform(-1./np.sqrt(n)/100, 1./np.sqrt(n)/100, (m+1, n)) for m, n in zip(nl[:-1], nl[1:])])

    err_mse = lambda X, bws, T: np.sum((calc_forward(X, bws)-T)**2)/2#/X.shape[0]

    get_random_matrix = lambda shape: np.random.random(shape)*2-1

    m = 5
    k = 3
    n = 4

    nl = [k, 6, n]

    X = get_random_matrix((m, k))
    bws1 = get_random_bws(nl)*100
    T = calc_forward(X, bws1)
    bws = get_random_bws(nl)

    errors = []
    eta = 0.001
    
    alpha = 0.2
    error_prev = err_mse(X, bws, T)
    bws_prev = deepcopy(bws)
    bwsd = calc_backprop(X, bws, T)
    bws = bws-bwsd*eta

    for i in xrange(0, 10000):
        bws_1 = bws+(bws-bws_prev)*alpha
        bwsd = calc_backprop(X, bws_1, T)
        bws_new = bws_1-bwsd*eta
        error = err_mse(X, bws_new, T)
        print("i: {}, error: {}".format(i, error))
        
        if error_prev < error:
            break

        error_prev = error
        bws = bws_new

        errors.append(error)

    print("bws1:\n{}".format(bws1))
    print("bws:\n{}".format(bws))

    plt.figure()
    plt.plot(errors)
    plt.show()

def advanced_gradient_descent_1_hidden_layer_train_valid_test():
    f_sig = lambda X: 1/(1+np.exp(-X))
    fd_sig = lambda X: f_sig(X)*(1-f_sig(X))
    f_tanh = lambda X: np.tanh(X)
    fd_tanh = lambda X: 1 - np.tanh(X)**2
    
    def calc_forward(X, bws):
        ones = np.ones((X.shape[0], 1))
        Y = X
        for bw in bws[:-1]:
            Y = f_tanh(np.hstack((ones, Y)).dot(bw))
        Y = f_sig(np.hstack((ones, Y)).dot(bws[-1]))

        return Y

    def calc_backprop(X, bws, T):
        Xs = []
        Ys = [X]
        ones = np.ones((X.shape[0], 1))
        Y = X
        for i, bw in zip(xrange(len(bws), 0, -1), bws[:-1]):
            A = np.hstack((ones, Y)).dot(bw); Xs.append(A)
            Y = f_tanh(A); Ys.append(Y)
        A = np.hstack((ones, Y)).dot(bws[-1]); Xs.append(A)
        Y = f_sig(A); Ys.append(Y)

        d = fd_sig(Xs[-1])*(Y-T)
        bwsd = deepcopy(bws)
        bwsd[-1] = np.hstack((ones, Ys[-2])).T.dot(d)
        
        for i in xrange(2, len(bws)+1):
            d = d.dot(bws[-i+1][1:].T)*fd_tanh(Xs[-i])
            bwsd[-i] = np.hstack((ones, Ys[-i-1])).T.dot(d)

        return bwsd

    def calc_backprop_num(X, bws, T):
        bwds = deepcopy(bws)
        epsilon = 0.0001

        for bw, bwd in zip(bws, bwds):
            y, x = bwd.shape
            for j in xrange(0, y):
                for i in xrange(0, x):
                    bw[j, i] += epsilon
                    f1 = err_mse(X, bws, T)
                    bw[j, i] -= epsilon*2
                    f2 = err_mse(X, bws, T)
                    bw[j, i] += epsilon

                    bwd[j, i] = (f1-f2)/2/epsilon

        return bwds
    
    def get_random_bws(nl):
        return np.array([np.random.uniform(-1./np.sqrt(n)/100, 1./np.sqrt(n)/100, (m+1, n)) for m, n in zip(nl[:-1], nl[1:])])

    err_mse = lambda X, bws, T: np.sum((calc_forward(X, bws)-T)**2)/2/X.shape[0]

    get_random_matrix = lambda shape: np.random.random(shape)*2-1

    m = 10
    k = 8
    n = 5

    nl = [k, 50, n]

    X_train = get_random_matrix((m, k))
    X_valid = get_random_matrix((int(m*0.4), k))
    X_test  = get_random_matrix((int(m*0.4), k))

    bws1 = get_random_bws(nl)*100
    T_train = calc_forward(X_train, bws1)+get_random_matrix((X_train.shape[0], n))*0.01
    T_valid = calc_forward(X_valid, bws1)+get_random_matrix((X_valid.shape[0], n))*0.01
    T_test = calc_forward(X_test, bws1)+get_random_matrix((X_test.shape[0], n))*0.01

    print("X_train.shape: {}".format(X_train.shape))
    print("T_train.shape: {}".format(T_train.shape))
    raw_input("ENTER!...")
    bws = get_random_bws(nl)

    errors_train = []
    errors_valid = []
    errors_test = []
    eta = 0.001
    
    alpha = 0.1
    error_train_prev = err_mse(X_train, bws, T_train)
    error_valid_prev = err_mse(X_valid, bws, T_valid)
    error_test_prev = err_mse(X_test, bws, T_test)
    bws_prev = deepcopy(bws)
    bwsd = calc_backprop(X_train, bws, T_train)
    bws = bws-bwsd*eta

    # TODO: finish the toy advanced gradient descent function!

    valid_bigger_times = 0

    form_green = Fore.GREEN+"{:>10.7f}"+Style.RESET_ALL
    form_magenta = Fore.MAGENTA+"{:>10.7f}"+Style.RESET_ALL
    form_cyan = Fore.CYAN+"{:>10.7f}"+Style.RESET_ALL

    format_row_1 = "i: {}, error_train: {}, error_valid: {}, error_test: {}".format(
        "{:4}", form_green, form_green, form_green)
        # "{:4}", form_green, form_magenta, form_cyan)
    format_row_2 = "       , diff_train:  {}, diff_valid:  {}, diff_test:  {}".format(
        form_cyan, form_cyan, form_cyan)
        # form_green, form_magenta, form_cyan)

    for i in xrange(0, 10000):
        bws_1 = bws+(bws-bws_prev)*alpha
        bwsd = calc_backprop(X_train, bws_1, T_train)
        bws_new = bws_1-bwsd*eta
        error_train = err_mse(X_train, bws_new, T_train)
        error_valid = err_mse(X_valid, bws_new, T_valid)
        error_test = err_mse(X_test, bws_new, T_test)

        print(format_row_1.format(
            i, error_train, error_valid, error_test))
        print(format_row_2.format(
            error_train_prev-error_train, error_valid_prev-error_valid, error_test_prev-error_test))
        
        # if error_valid_prev < error_valid:
        #     valid_bigger_times += 1
        #     if valid_bigger_times > 2000:
        #         break
        # else:
        #     valid_bigger_times = 0

        error_valid_prev = error_valid
        bws_prev = bws
        bws = bws_new

        errors_train.append(error_train)
        errors_valid.append(error_valid)
        errors_test.append(error_test)

    # print("bws1:\n{}".format(bws1))
    # print("bws:\n{}".format(bws))

    plt.figure()
    plt.plot(errors_train, "b-")
    plt.plot(errors_valid, "g-")
    plt.plot(errors_test, "r-")
    plt.show()

if __name__ == "__main__":
    # simple_grad_calculating()
    # simple_gradient_descent()
    # simple_gradient_descent_1_hidden_layer()
    simple_gradient_descent_1_hidden_layer_eta_checker()
    # advanced_gradient_descent_1_hidden_layer()
    # advanced_gradient_descent_1_hidden_layer_train_valid_test()

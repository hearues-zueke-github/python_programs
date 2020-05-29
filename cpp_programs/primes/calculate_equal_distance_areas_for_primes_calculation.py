#! /usr/bin/python3

import dill
import gzip
import os
import sys

import numpy as np

import matplotlib.pyplot as plt

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

def calc_error_sum(ns_arr):
    areas = 2/3*(ns_arr[1:]**(3/2)-ns_arr[:-1]**(3/2))
    areas_diff = np.diff(areas)
    error_sum = np.sum(areas_diff**2)*(1/2)
    return error_sum


def calc_best_fitting_spacing(n, n_s, n_e, ns_arr=None):
    if ns_arr is None:
        ns_arr = n_s+(n_e-n_s)/n*np.arange(0, n+1)
    elif isinstance(ns_arr, np.ndarray) and len(ns_arr.shape)==1 and ns_arr.shape[0]==n+1:
        ns_arr = ns_arr.copy()
    else:
        assert False

    lambd = 0.0001


    def calc_new_ns_arr_and_error(ns_arr, lambd2):
        areas = 2/3*(ns_arr[1:]**(3/2)-ns_arr[:-1]**(3/2))
        areas_diff = np.diff(areas)

        areas_deriv_0 = 2*areas_diff*(2*3/2*ns_arr[1:-1]**(1/2))
        areas_deriv_left = np.hstack(((0, ), 2*areas_diff[:-1]*(-3/2*ns_arr[2:-1]**(1/2))))
        areas_deriv_right = np.hstack((2*areas_diff[1:]*(-3/2*ns_arr[1:-2]**(1/2)), (0, )))

        areas_deriv = areas_deriv_0+areas_deriv_left+areas_deriv_right

        areas_deriv_calc = np.sqrt(np.abs(areas_deriv))*np.sign(areas_deriv)
        ns_arr_new = ns_arr+lambd2*np.hstack(((0, ), areas_deriv_calc, (0, )))

        error_sum = calc_error_sum(ns_arr_new)

        return ns_arr_new, error_sum
    

    ns_arr, error_sum = calc_new_ns_arr_and_error(ns_arr, lambd)
    # print("error_sum at beginning: {}".format(error_sum))

    l_error_sum_best = [error_sum]
    l_lambd_best = [lambd]
    for iter_round in range(1, 2500):
        l_lambd = []
        l_errors = []
        l_ns_arr = []

        ns_arr_new, error_sum_new = calc_new_ns_arr_and_error(ns_arr, lambd)
        l_lambd.append(lambd)
        l_errors.append(error_sum_new)
        l_ns_arr.append(ns_arr_new)

        lambd_pos = lambd
        lambd_neg = lambd
        for i in range(0, 10):
            lambd_pos *= 1.3
            ns_arr_new, error_sum_new = calc_new_ns_arr_and_error(ns_arr, lambd_pos)
            l_lambd.append(lambd_pos)
            l_errors.append(error_sum_new)
            l_ns_arr.append(ns_arr_new)

            lambd_neg *= 0.3
            ns_arr_new, error_sum_new = calc_new_ns_arr_and_error(ns_arr, lambd_neg)
            l_lambd.append(lambd_neg)
            l_errors.append(error_sum_new)
            l_ns_arr.append(ns_arr_new)

        best_i = np.argmin(l_errors)
        lambd = l_lambd[best_i]
        error_sum = l_errors[best_i]
        ns_arr = l_ns_arr[best_i]

        l_error_sum_best.append(error_sum)
        l_lambd_best.append(lambd)

        # print("iter_round: {:5}, error_sum: {}".format(iter_round, error_sum))
    # print("error_sum at end: {}".format(error_sum))

    return ns_arr, l_error_sum_best, l_lambd_best


if __name__=='__main__':
    n_s = 100000
    n_e = 1000000
    
    PATH_OBJ_FILE_ABS = PATH_ROOT_DIR+'d_objs_n_s_{}_n_e_{}.pkl.gz'.format(n_s, n_e)
    if not os.path.exists(PATH_OBJ_FILE_ABS):
        d_objs = {}
    else:
        with gzip.open(PATH_OBJ_FILE_ABS, 'rb') as f:
            d_objs = dill.load(f)

    l_spacings = []
    # d_error_lambd = {}
    for n in range(1, 5):
        t = (n_s, n_e, n)
        if not t in d_objs:
            n_proc = np.arange(1, n)/n 
            ns_arr = np.hstack(((n_s, ), (n_proc*n_e**(3/2)+n_s**(3/2)*(1-n_proc))**(2/3), (n_e, )))
            # ns_arr, l_error_sum_best, l_lambd_best = calc_best_fitting_spacing(n, n_s, n_e)
            # d_error_lambd[n] = (l_error_sum_best, l_lambd_best)
            error_sum = calc_error_sum(ns_arr)
            d_objs[t] = (ns_arr, error_sum)
        else:
            ns_arr_old, error_sum_old = d_objs[t]
            if error_sum_old>=10**-7:
                ns_arr, l_error_sum_best, l_lambd_best = calc_best_fitting_spacing(n, n_s, n_e, ns_arr=ns_arr_old)
                # d_error_lambd[n] = (l_error_sum_best, l_lambd_best)
                error_sum = calc_error_sum(ns_arr)

                if not np.any(np.isnan(ns_arr)) and error_sum<error_sum_old:
                    d_objs[t] = (ns_arr, error_sum)
            else:
                ns_arr = ns_arr_old
                error_sum = error_sum_old

        # print("n: {}, error_sum: {}".format(n, error_sum))
        print("n: {}, ns_arr: {}".format(n, ns_arr))

        # print("Final:")
        # print("ns_arr: {}".format(ns_arr))
        areas = 2/3*(ns_arr[1:]**(3/2)-ns_arr[:-1]**(3/2))
        areas_diff = np.diff(areas)
        # print("areas_diff: {}".format(areas_diff))

        l_spacings.append((n, ns_arr))

    plt.figure()

    plt.title('Spacing, Plot')
    plt.xlabel('0 to 1 spacings')
    plt.ylabel('Amount of spaces')

    def random_color_hex_string():
        arr = np.random.randint(0, 128, (3, ))
        return ''.join([hex(i)[2:].upper().zfill(2) for i in arr])

    for n, ns_arr in l_spacings:
        # plt.plot(np.cumsum([0]+[1/(len(ns_arr)-1)]*(len(ns_arr)-1)), ns_arr, '.', color='#'+random_color_hex_string())
        plt.plot(ns_arr, [n]*ns_arr.shape[0], '.', color='#'+random_color_hex_string())

    plt.show(block=False)

    with gzip.open(PATH_OBJ_FILE_ABS, 'wb') as f:
        dill.dump(d_objs, f)

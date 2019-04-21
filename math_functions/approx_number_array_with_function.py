#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import gzip
import os
import sys

import numpy as np

from math import gcd
from math import factorial as fac

import decimal
decimal.getcontext().prec = 400
from decimal import Decimal as Dec
Decimal = Dec

import matplotlib.pyplot as plt

path_dir_root = os.path.dirname(os.path.abspath(__file__))+"/"

sys.path.append("../math_functions/")
import vector_utils

def find_best_divisible_whole_numbers(x):
    # x = Dec("2.375")
    # x = Dec("0.916666666666666666666666666")
    # x = Dec("2.583333333333333333333333333333333333333333333333333")
    is_neg = x<0
    x_pos = x
    if is_neg:
        x_pos *= -1
    x_comma = x_pos%1
    x_whole = x_pos//1
    # print("x_comma: {}".format(x_comma))
    # print("x_whole: {}".format(x_whole))

    str_comma = str("{:.400f}".format(x_comma)).split(".")[1]

    best_divident = Dec("1")
    best_divisor = Dec("1")
    best_error = np.abs(x_comma-best_divident/best_divisor)
    # print("x: {}".format(x))

    # print("str_comma: {}".format(str_comma))
    for idx2 in range(1, 250):
        for idx1 in range(0, idx2):
            str_comma_left = str_comma[:idx1]
            str_comma_right = str_comma[idx1:idx2]

            i_left = int(str_comma_left) if str_comma_left != "" else 0
            i_right = int(str_comma_right)

            # print("idx1: {}, idx2: {}".format(idx1, idx2))
            # print("\nstr_comma_left: {}, str_comma_right: {}".format(str_comma_left, str_comma_right))
            # print("i_left: {}, i_right: {}".format(i_left, i_right))

            divisor_9s = 10**(idx2-idx1)-1
            divident = i_left*divisor_9s+i_right
            divisor = divisor_9s*10**idx1

            common_div = gcd(divident, divisor)
            divident //= common_div
            divisor //= common_div
            # print("divident: {}".format(divident))
            # print("divisor: {}".format(divisor))

            d = Dec(divident)/Dec(divisor)
            # print("d: {}".format(d))

            error = np.abs(x_comma-d)
            # print("divident: {}, divisor: {}, error: {}".format(divident, divisor, error))
            # print("x_comma:\n{}".format(x_comma))
            # print("d:\n{}".format(d))
            if error < best_error:
                best_divident = divident
                best_divisor = Decimal(divisor)
                best_error = error
                # print("divident: {}, divisor: {}, error: {}".format(divident, divisor, error))

    return (-1)**is_neg*(x_whole*best_divisor+best_divident), best_divisor

    if False: # First approach
        a = Dec("0")
        b = Dec("1")

        diff_prev = np.abs(x-a/b)
        best_a = a
        best_b = b

        for i in range(0, 1000000):
            if a/b > x_comma:
                b += 1
            elif a/b < x_comma:
                a += 1
            # print("i: {}".format(i))
            # print("a: {}, b: {}, a/b: {}".format(a, b, a/b))
            diff = np.abs(x_comma-a/b)
            # print("diff: {}".format(diff))
            if diff_prev > diff:
                diff_prev = diff
                best_a = a
                best_b = b
                print("a: {}, b: {}".format(a, b))
            # print("")
        print("best_a: {}, best_b: {}".format(best_a, best_b))
        best_a += x_whole*best_b
        return ((-1)**is_neg*best_a, best_b)


def find_best_fitting_functions_for_random_sequence_quality_arrays():
    path_file_obj = path_dir_root+"../sequence_generators/obj.pkl.gz"
    with gzip.open(path_file_obj, "rb") as fout:
        objs_dict = dill.load(fout)
    values_tabl_orig = objs_dict["values_tabl_orig"]
    values_tabl = objs_dict["values_tabl"]

    globals()["values_tabl_orig"] = values_tabl_orig
    globals()["values_tabl"] = values_tabl

    print("values_tabl_orig.shape: {}".format(values_tabl_orig.shape))
    print("values_tabl.shape: {}".format(values_tabl.shape))

    # n's for row_idx:
    """
    [
      (0,  0),
      (1,  2),
      (2,  4),
      (3,  5),
      (4,  7),
      (5,  9),
      (6,  11),
      (7,  13),
      (8,  15),
      (9,  17),
      (10, 19),
      (11, 21),
      (12, 23),
      (13, 25),
      (14, 27),
      (15, 29),
      (16, 31),
      (17, 33),
      (18, 35),
      (19, 37),
      etc...
    ]
    """

    row_idx = 15
    n = 29

    xs_all = np.array([Dec(int(x)) for x in np.arange(1, values_tabl.shape[1]+1)])
    ys_all = np.array([Dec(y) for y in values_tabl[row_idx]])

    """
    for values_tabl[1]:
        a_1 = np.array([
            Dec('1'),
            Dec('3')/Dec('2'),
            Dec('1')/Dec('2')
        ])
    for values_tabl[2]:
        a_2 = np.array([
            Dec('1'),
            Dec('31')/Dec('12'),
            Dec('19')/Dec('8'),
            Dec('11')/Dec('12'),
            Dec('1')/Dec('8')
        ])
    for values_tabl[3]:
        a_3 = np.array([
            Dec('1'),
            Dec('49')/Dec('12'),
            Dec('45')/Dec('8'),
            Dec('10')/Dec('3'),
            Dec('7')/Dec('8'),
            Dec('1')/Dec('12')
        ])
    for values_tabl[4]:
        a_4 = np.array([
            Dec('1'),
            Dec('79')/Dec('20'),
            Dec('589')/Dec('72'),
            Dec('229')/Dec('24'),
            Dec('863')/Dec('144'),
            Dec('159')/Dec('80'),
            Dec('47')/Dec('144'),
            Dec('1')/Dec('48'),
        ])
    for values_tabl[5]:
        a_5 = np.array([
            Decimal('1')/Decimal('1'),
            Decimal('67')/Decimal('10'),
            Decimal('10021')/Decimal('720'),
            Decimal('5821')/Decimal('360'),
            Decimal('2597')/Decimal('192'),
            Decimal('3829')/Decimal('480'),
            Decimal('477')/Decimal('160'),
            Decimal('13')/Decimal('20'),
            Decimal('43')/Decimal('576'),
            Decimal('1')/Decimal('288'),
        ])
    for values_tabl[6]:
        a_6 = np.array([
            Decimal('1')/Decimal('1'),
            Decimal('549')/Decimal('70'),
            Decimal('2129')/Decimal('90'),
            Decimal('98803')/Decimal('2880'),
            Decimal('4641')/Decimal('160'),
            Decimal('22741')/Decimal('1280'),
            Decimal('25727')/Decimal('2880'),
            Decimal('64987')/Decimal('18878'),
            Decimal('643')/Decimal('720'),
            Decimal('1627')/Decimal('11520'),
            Decimal('7')/Decimal('576'),
            Decimal('1')/Decimal('2304')
        ])
    for values_tabl[7]:
        a_7 = np.array([
            Decimal('1')/Decimal('1'),
            Decimal('-4489')/Decimal('280'),
            Decimal('-23281')/Decimal('2520'),
            Decimal('159043')/Decimal('2688'),
            Decimal('3472619')/Decimal('34560'),
            Decimal('335005')/Decimal('4608'),
            Decimal('4377689')/Decimal('138240'),
            Decimal('86299')/Decimal('8064'),
            Decimal('1098137')/Decimal('322560'),
            Decimal('25247')/Decimal('26880'),
            Decimal('25483')/Decimal('138240'),
            Decimal('13')/Decimal('576'),
            Decimal('211')/Decimal('138240'),
            Decimal('1')/Decimal('23040'),
            ])
    for values_tabl[8]:
        a_8 = np.array([
            Decimal('1')/Decimal('1'),
            Decimal('428221')/Decimal('2520'),
            Decimal('10153193')/Decimal('50400'),
            Decimal('-13979831')/Decimal('181440'),
            Decimal('-12849299')/Decimal('120960'),
            Decimal('21008261')/Decimal('241920'),
            Decimal('15858077')/Decimal('115200'),
            Decimal('3446545')/Decimal('48384'),
            Decimal('40052759')/Decimal('1935360'),
            Decimal('25327651')/Decimal('5806080'),
            Decimal('8657071')/Decimal('9676800'),
            Decimal('117679')/Decimal('645120'),
            Decimal('175')/Decimal('6144'),
            Decimal('781')/Decimal('276480'),
            Decimal('43')/Decimal('276480'),
            Decimal('1')/Decimal('276480'),
        ])
    """

    move_idx = 0
    xs = xs_all[move_idx:move_idx+n+1]
    ys = ys_all[move_idx:move_idx+n+1]
    X = xs.reshape((-1, 1))**np.arange(0, n+1)
    X_all = xs_all.reshape((-1, 1))**np.arange(0, n+1)
    globals()["xs"] = xs
    globals()["ys"] = ys
    globals()["X"] = X

    print("xs: {}".format(xs))
    print("ys: {}".format(ys))
    print("X:\n{}".format(X))

    # e.g. quadratic function
    # a = np.array([Dec(i) for i in (np.random.random((n+1, ))-0.5)*3])
    # a = np.array([Dec(1) for _ in range(0, n+1)])
    
    a = np.array(
        [Decimal("1")/Decimal(10**i) for i in range(0, n+1)]
    )

   #  a = np.array(
   # [
   #      Dec('1'),
   #      Dec('11')/Dec('12'),
   #      Dec('19')/Dec('8'),
   #      Dec('31')/Dec('12'),
   #      Dec('1')/Dec('8')
   #  ]
   #  )

    # alpha = Dec(0.000005)*np.array([Dec("1") / Dec(fac(i)) for i in range(1, n+2)])
    alpha = Dec(0.00005)*(Dec("0.005"))**np.arange(1, n+2)
    # alpha = Dec(0.00005)*(Dec("1")/values_tabl.shape[1]/20)**np.arange(1, n+2)
    alpha_inc = Dec("1.1")
    alpha_dec = Dec("1.5")
    print("alpha: {}".format(alpha))

    f = lambda a: X_all.dot(a)

    i = 0
    # for i in range(1, 100001):
    yfs = f(a)
    err = ys_all-yfs
    err_val_prev = np.sum(err**2)
    err_val_10000 = err_val_prev
    err_diff_10000 = 1
    err_diff_10000_prev = 1


    if False:
        X_no_diag = X.copy()
        X_no_diag[(np.arange(0, n+1), np.arange(0, n+1))] = Decimal(0)
        print("X_no_diag:\n{}".format(X_no_diag))
        X_diag = np.diag(X)
        print("X_diag:\n{}".format(X_diag))

        # eigenvalues
        w = np.linalg.eig(((np.diag(1/np.diag(X))).dot(X-np.diag(np.diag(X)))).astype(np.float))[0]
        max_w = np.max(np.abs(w))
        print("w: {}".format(w))
        print("max_w: {}".format(max_w))

    # i = 0
    # print("i: {}, a: {}".format(i, a))
    # while True:
    #     i += 1
    #     a_new = Decimal(1)/X_diag*(ys-X_no_diag.dot(a))
    #     a = a_new
    #     a[0] = Dec(1)
    #     print("i: {}, a: {}".format(i, a))
    #     input("ENTER...")
    # return

    a = vector_utils.inv_with_decimal(X).dot(ys)

    f = lambda a: X_all.dot(a)
    yfs = f(a)
    err = ys_all-yfs

    globals()["a"] = a
    a_fractions = [find_best_divisible_whole_numbers(x) for x in a]
    globals()["a_fractions"] = a_fractions
    # a = np.array([Decimal(x) for x in np.linalg.inv(X).dot(ys)])
    print("a:\n{}".format(a))
    print("a_fractions:\n{}".format(str(a_fractions).replace("[", "[\n").replace("]", "\n]").replace(")), ", ")),\n")))
    print("err: {}".format(err))
    return


    fractions_of_a = [find_best_divisible_whole_numbers(x) for x in a]
    print("fractions_of_a:\n{}".format(fractions_of_a))
    print("    a_{} = np.array([".format(row_idx))
    for frac, ai in zip(fractions_of_a, a):
        print("        Decimal('{}')/Decimal('{}'),".format(str(frac[0]), str(frac[1])))
    print("    ])")

    # print("a_9-a:\n{}".format(a_9-a))
    # return

    a_true = np.array([frac[0]/frac[1] for frac in fractions_of_a])

    # print("a-a_true: {}".format(a-a_true))

    a_idx_same = [0, 1, 2, 6, 7, 8, 9]
    is_idx_fix = False#True
    while True:
        i += 1

        yfs = f(a)
        err = ys_all-yfs
        da = X_all.T.dot(err)
        
        a_new = a+alpha*da
        if is_idx_fix:
            a_new[a_idx_same] = a[a_idx_same]

        yfs = f(a_new)
        err = ys_all-yfs
        err_val = np.sum(err**2)

        j = 1
        if err_val >= err_val_prev:
            alpha_new = alpha
            if is_idx_fix:
                a_new[a_idx_same] = a[a_idx_same]
            while err_val >= err_val_prev:
                j += 1
                alpha_new /= alpha_dec
                a_new = a+alpha_new*da
                if is_idx_fix:
                    a_new[a_idx_same] = a[a_idx_same]

                yfs = f(a_new)
                err = ys_all-yfs
                err_val = np.sum(err**2)
                if j > 5:
                    break
        else:
            alpha_new = alpha
            if is_idx_fix:
                a_new[a_idx_same] = a[a_idx_same]
            while err_val < err_val_prev:
                j += 1
                alpha_new *= alpha_inc
                a_new = a+alpha_new*da
                if is_idx_fix:
                    a_new[a_idx_same] = a[a_idx_same]

                yfs = f(a_new)
                err = ys_all-yfs
                err_val = np.sum(err**2)
                if j > 10:
                    break

            alpha_new /= alpha_inc
            a_new = a+alpha_new*da
            if is_idx_fix:
                a_new[a_idx_same] = a[a_idx_same]

            yfs = f(a_new)
            err = ys_all-yfs
            err_val = np.sum(err**2)

        alpha = alpha_new
        yfs_new = f(a_new)
        err_new = ys_all-yfs_new
        err_val = np.sum(err_new**2)

        a = a_new
        if i % 1000 == 0:
            print("i: {i}, j: {j}, err_val: {err_val}".format(i=i, j=j, err_val=err_val))
            # print("err_val_10000-err_val: {}".format(err_val_10000-err_val))
            err_diff_10000_prev = err_diff_10000
            err_diff_10000 = err_val_10000-err_val
            err_val_10000 = err_val
            percentage_performance = err_diff_10000/err_val_10000
            # print("err_val_10000: {}".format(err_val_10000))
            # print("err_diff_10000: {}".format(err_diff_10000))
            # print("err_diff_10000_prev: {}".format(err_diff_10000_prev))
            print("percentage_performance: {}%".format(percentage_performance*100))
            print("a:\n{a}".format(a=str(a.tolist()).replace(", ", ",\n")))

        err_val_prev = err_val

        if i % 10000 == 0 and False:
            inp = input("What to do next?\n")
            if "+" in inp:
                lst = inp.split("+")
                a[int(lst[0])] += Dec(lst[1])
            elif "-" in inp:
                lst = inp.split("-")
                a[int(lst[0])] -= Dec(lst[1])
            print("a: {}".format(a))
            input("ENTER...")

    # print("final: yfs: {}".format(yfs))

    # plt.figure()

    # plt.title("Values of values_tabl per row (multipliers of values_tabl_orig 1st row values)")

    # xmax = values_tabl.shape[1]
    # xs = np.arange(1, xmax+1)
    # plt.xlim([0.5, xmax+0.5])
    # plt.ylim([0, 100])

    # ps = []
    # ps_text = []
    # for i in range(0, 5):
    #     ps.append(plt.plot(xs, values_tabl[i])[0])
    #     ps_text.append("Row nr. {row}".format(row=i+1))

    # plt.legend(ps, ps_text)

    # plt.show()


# approx a given list of numbers with a polynomial function e.g. and find the
# best possible fitting possible, with the minimum n-th grade of polynomial function
if __name__ == "__main__":
    find_best_fitting_functions_for_random_sequence_quality_arrays()

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
from functools import reduce

from PIL import Image

import numpy as np

from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

from utils_modulo_sequences import prettyprint_dict

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

from PIL import ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style


def time_measure(f, args):
    start_time = time()
    ret = f(*args)
    end_time = time()
    diff_time = end_time-start_time
    return ret, diff_time


class ShowImg(Frame, object):
    def __init__(self, img):
        parent = Tk()
        Frame.__init__(self, parent)
        self.pack(fill=BOTH, expand=1)
        label1 = Label(self)
        label1.photo= ImageTk.PhotoImage(img)
        label1.config(image=label1.photo)
        label1.pack(fill=BOTH, expand=1)
        parent.mainloop()


def do_cycles(a, c, m):    
    vals = np.zeros((m, ), dtype=np.int)
    x = 0
    x = (a*x+c)%m
    for i in range(1, m):
        x = (a*x+c)%m
        vals[i] = x
    return vals.tolist()


def get_img(arr, m=0):
    max_val = m if m != 0 else np.max(arr)
    arr = arr.astype(np.float64)/(max_val+1)
    pix = (arr*255.999).astype(np.uint8)
    img = Image.fromarray(pix)
    return img


def get_full_cycles(m):
    vals = np.arange(0, m)
    zeros = np.zeros((m, ), dtype=np.int)
    full_cycles = {}
    arr = np.zeros((m, m), dtype=np.int)

    # all_arrs = np.zeros((m**2, m+2), dtype=np.int)
    for i, a in enumerate(vals, 0):
        x_iter = vals
        # x_iter = zeros
        # x_iter = ((a*x_iter)+vals)%m
        arr[:, 0] = x_iter
        for l in range(1, m, 1):
            x_iter = ((a*x_iter)+vals)%m
            arr[:, l] = x_iter
        for k, row in enumerate(arr, 0):
            t = (a, k)
            if np.unique(row).shape[0]==m:
                full_cycles[t] = np.roll(row, 1).tolist()
    return full_cycles


def get_full_cycles_faster(m):
    zeros = np.zeros((m**2, ), dtype=np.int)
    full_cycles = {}
    arr = np.zeros((m**2, m+1), dtype=np.int)

    asz = np.zeros((m, m), dtype=np.int)
    asz[:] = np.arange(0, m).reshape((-1, 1))
    cs = np.zeros((m, m), dtype=np.int)
    cs[:] = np.arange(0, m).reshape((1, -1))

    asz = asz.reshape((-1, ))
    cs = cs.reshape((-1, ))

    x_iter = zeros
    for l in range(1, m+1, 1):
        x_iter = ((asz*x_iter)+cs)%m
        arr[:, l] = x_iter
    arr_sort = np.sort(arr, axis=1)
    idxs = np.all((arr_sort[:, 2:]-arr_sort[:, 1:-1])==1, axis=1)

    for a, c, row in zip(asz[idxs], cs[idxs], arr[idxs]):
        t = (a, c)
        if np.unique(row).shape[0]==m:
            full_cycles[t] = row[:-1].tolist()
    return full_cycles


def get_full_cycles_amount(m):
    print("m: {}".format(m))
    rs=lambda x,t: x.reshape(t)
    d={'dtype':np.int32};zr=np.zeros;ar=np.arange
    z=zr((m,m),**d);r=ar(0,m,**d);j=ar(0,m**2)
    c=z+r;a=rs(c.T,(-1,));c=rs(c,(-1,))
    cy=zr((m**2,m),**d);b=zr((m**2,m),**d);x=zr((m**2,),**d)
    x=(a*x+c)%m;b[j,x]+=1;cy[j,0]=x;cl=zr((m,),**d)
    for i in range(0, m-1):
        x=(a*x+c)%m;b[j,x]+=1;cy[j,i+1]=x;idxs=np.all(b<2,axis=1)
        s=np.sum(~idxs);cl[i]=s
        if s==0: continue
        a=a[idxs];x=x[idxs];c=c[idxs];b=b[idxs];cy=cy[idxs];j=j[:-s]
    cl[-1]=c.shape[0]
    return a, c, cy, cl


def get_mod_max_cycles_amount(m):
    rs=lambda x,t: x.reshape(t)
    d={'dtype':np.int32};zr=np.zeros;ar=np.arange
    z=zr((m,m),**d);r=ar(0,m,**d);j=ar(0,m**2)
    c=rs(z+r,(-1,));a=rs(z+rs(r,(-1,1)),(-1,))
    b=zr((m**2,m),**d);x=zr((m**2,),**d)
    x=(a*x+c)%m;b[j,x]+=1
    for i in range(0, m-1):
        x=(a*x+c)%m;b[j,x]+=1;idxs=np.all(b<2,axis=1)
        s=np.sum(~idxs)
        if s==0: continue
        a=a[idxs];x=x[idxs];c=c[idxs];b=b[idxs];j=j[:-s]
    return c.shape[0]


def find_new_cycle_lens():
    file_path = "num_mod_lens.pkl.gz"
    def get_empty_dict():
        obj = {'lens': [], 'ms': []}
        obj['last_m'] = 1
        return obj
    print("Loading obj from file_path: {}".format(file_path))
    obj = get_pkl_gz_obj(get_empty_dict, file_path)
    # print("read: obj: {}".format(obj))

    next_m = obj['last_m']
    for m in range(next_m, next_m+10):
        print("m: {}".format(m))
        full_cycles = get_full_cycles_faster(m)
        length = len(full_cycles)

        obj['ms'].append(m)
        obj['lens'].append(length)

    obj['last_m'] = next_m+10

    # print("finished: obj: {}".format(obj))

    print("Savinbg obj at file_path: {}".format(file_path))
    save_pkl_gz_obj(obj, file_path)

    lens = obj['lens']
    print("lens:\n{}".format(lens))

    u, c = np.unique(lens, return_counts=True)

    print("u:\n{}".format(u))
    print("c:\n{}".format(c))

    return lens


def get_max_cycle_lengths(nn):
    def a(n):
        i32=np.int32;t=(-1,1);d={'dtype':i32}
        c=np.arange(0,n,**d)
        a=(np.zeros((n, n),**d)+c).reshape(t)
        x=(np.zeros((n**2,n),**d)+c).T.reshape((n**2, n))
        cl=np.zeros((n,),**d)
        x_i=x;b_0=(x==n)
        for i in range(0, n):
            x_i=(a*x_i+c)%n;b_i=(x_i==x)&(~b_0)
            b_0|=b_i;cl[i]=np.sum(b_i)
        return cl[-1]//n
    return [a(n) for n in range(1,nn+1)]


def get_cycle_cl(n):
    t=(-1,1);d={'dtype':np.int32}
    c=np.arange(0,n,**d)
    a=(np.zeros((n, n),**d)+c).reshape(t)
    x=(np.zeros((n**2,n),**d)+c).T.reshape((n**2, n))
    cl=np.zeros((n,),**d)
    x_i=x;b_0=(x==n)
    for i in range(0, n):
        x_i=(a*x_i+c)%n;b_i=(x_i==x)&(~b_0)
        b_0|=b_i;cl[i]=np.sum(b_i)
    return cl


if __name__ == "__main__":
    # lens = find_new_cycle_lens()
    # sys.exit(0)

    nn = int(sys.argv[1])

    # lst_max_cycles = []
    # lst_amounts_a_c = []
    # lst_amounts_a_bigger_1 = []

    # for m in range(1, nn+1):
    #     arr = np.array(list(get_full_cycles_faster(m).keys())).T
    #     lst_max_cycles.append(arr.shape[1])
    #     amount_a = np.unique(arr[0]).shape[0]
    #     amount_c = np.unique(arr[1]).shape[0]
    #     lst_amounts_a_c.append((m, amount_a, amount_c))
    #     if amount_a > 1:
    #         lst_amounts_a_bigger_1.append((m, amount_a, amount_c))

    # print("lst_max_cycles: {}".format(lst_max_cycles))
    # # print("lst_amounts_a_c: {}".format(lst_amounts_a_c))
    # # print("lst_amounts_a_bigger_1: {}".format(lst_amounts_a_bigger_1))

    # lst_amounts_c = reduce(lambda a, b: a+[b[2]], lst_amounts_a_c, [])
    # print("lst_amounts_c: {}".format(lst_amounts_c))

    # sys.exit(0)

    lista = []
    lista_sum = []
    lista_sum_normalized = []
    list_num_max_cycles = []
    list_cls = []
    rs=lambda x,t: x.reshape(t)
    d={'dtype':np.int32};zr=np.zeros;ar=np.arange
    for m in range(1, nn+1):
        cl = get_cycle_cl(m)
        list_num_max_cycles.append(cl[-1]//m)
        
        print("m: {}".format(m))
        lista.append(np.sum(cl>0))
        lista_sum.append(np.sum(cl))
        lista_sum_normalized.append(np.sum(cl)//m)

        list_cls.append(cl)
    print("lista: {}".format(lista))
    print("lista_sum: {}".format(lista_sum))
    print("lista_sum_normalized: {}".format(lista_sum_normalized))
    print("list_num_max_cycles: {}".format(list_num_max_cycles))

    # find the max of list_cls values
    # find the normalizes list: e.g. [9, 6, 6,] // 3 -> [3, 2, 2] -> max=3

    lnmc = np.array(list_num_max_cycles)

    list_norm_cl_max = [np.max(l)//reduce(lambda a, b: np.gcd(a, b), (lambda x: x[x>0])(l)) for l in list_cls]
    print("list_norm_cl_max: {}".format(list_norm_cl_max))
    l = np.array(list_norm_cl_max)

    # ranges = [[1]]+[lnmc[2**i:2**(i+1)] for i in range(0, 7)]
    # min_max_2_pow_n_ranges = [(np.min(r), np.max(r)) for r in ranges]
    # print("min_max_2_pow_n_ranges: {}".format(min_max_2_pow_n_ranges))

    # nn = int(sys.argv[1])
    # lista = get_max_cycle_lengths(nn)
    # print("lista: {}".format(lista))

    # [np.sum(get_cycle_cl(i))//i for i in range(1, 21)] = A279912(n) * A000027(n) sequence Cf.
    # list_num_max_cycles like: A308828(n) = A135616(n) / A000027(n) sequence Cf.
    ### lista_sum_normalized like = A279912(n) * A000027(n) sequence Cf.

    sys.exit(0)

    def get_empty_object():
        return {'nn': 1, 'lst': []}
    file_path="num_mod_lens_2.pkl.gz"
    obj = get_pkl_gz_obj(get_empty_object, file_path)
    lst = obj['lst']
    nn_old = obj['nn']
    nn_new = nn_old+0
    obj['nn'] = nn_new
    lst2 = [get_full_cycles_amount(m) for m in range(nn_old, nn_new)]
    lst += lst2
    
    # save_pkl_gz_obj(obj, file_path)
    
    sys.exit(0)

    lsts_a, lsts_c, lsts_cy, lsts_cl = list(zip(*lst))

    lst_unique_amount_a = [np.unique(a).shape[0] for a in lsts_a]
    lst_unique_amount_c = [np.unique(c).shape[0] for c in lsts_c] # A000010
    a = np.array(lst_unique_amount_a)
    c = np.array(lst_unique_amount_c)

    ua = np.unique(a)
    def div_by_first_value(arr):
        return arr//arr[0]
    lsts_of_a = [(i, )+(lambda x: (x, div_by_first_value(x)))(np.where(a==i)[0]+1) for i in ua]
    print("lsts_of_a:")
    for i, lst_a, lst_a_norm in lsts_of_a:
        print("i: {}, lst_a: {}, lst_a_norm: {}".format(i, lst_a, lst_a_norm))

    uc = np.unique(c)
    lsts_of_c = [(i, )+(lambda x: (x, ))(np.where(c==i)[0]+1) for i in uc]
    print("lsts_of_c:")
    for i, lst_c in lsts_of_c:
        print("i: {}, lst_c: {}".format(i, lst_c))

    multiples_a = [l for _, l, _ in lsts_of_a][1:]
    only_one_val_per_a = np.array([_[0] for _ in multiples_a])
    only_one_val_per_a_orig = only_one_val_per_a.copy()
    idxs_orig = np.arange(0, only_one_val_per_a.shape[0])

    combine_idxs = []
    for i, val_a in enumerate(only_one_val_per_a):
        if only_one_val_per_a.shape[0] == 0:
            break
        idxs = only_one_val_per_a%val_a==0
        idxs[idxs_orig==i] = False
        combine_idxs.append((i, np.where(idxs)[0].tolist()))
        # only_one_val_per_a = only_one_val_per_a[~idxs]
        # idxs_orig = idxs_orig[~idxs]
    print("multiples_a: {}".format(multiples_a))
    print("combine_idxs: {}".format(combine_idxs))

    for i, arr in combine_idxs:
        if len(arr) == 0:
            continue
        print("{}: {}".format(only_one_val_per_a_orig[i], list(map(lambda x: only_one_val_per_a_orig[x], arr))))
        arr2 = np.sort(np.hstack([multiples_a[i]]+[multiples_a[j] for j in arr]))
        print("arr2: {}".format((arr2//arr2[0]).tolist()))

    # lsts_of_a[0][1].tolist() = A078779

    # lst_unique_amount_a(n) could be an original sequence!
    # A000010(n) = A308828(n) / lst_unique_amount_a(n)

    # print("lst_unique_amount_a: {}".format(lst_unique_amount_a))
    # print("lst_unique_amount_c: {}".format(lst_unique_amount_c))



    sys.exit(0)

    nn = int(sys.argv[1])
    lens = []
    length_m_dict = {}
    # for m in get_primes(1000):
    for m in range(1,  nn+1, 1):
        full_cycles = get_full_cycles_faster(m)
        lens.append(len(full_cycles))

        length = len(full_cycles)

        if not length in length_m_dict:
            length_m_dict[length] = []
        length_m_dict[length].append(m)

        if 1:
            cycles_lst = [(a, c, cycle) for (a, c), cycle in full_cycles.items()]
            print("cycles_lst:")
            for i, (a, c, cycle) in enumerate(cycles_lst, 0):
                print("i: {}, {}: {}".format(i, (a, c), cycle))

    print("lens: {}".format(lens))
    sys.exit(0)

    # a_c_factors = list(full_cycles.keys())
    # print("a_c_factors: {}".format(a_c_factors))

    def get_all_possible_cycles(full_cycles, m):
        cycle_factors_dict = {tuple(cycle): (((a, c), ), ) for (a, c), cycle in full_cycles.items()}
        factors_cycle_dict = {v: k for k, v in cycle_factors_dict.items()}

        for iteration in range(1, 10):
            print("iteration: {}".format(iteration))
            a_c_factors = list(cycle_factors_dict.values())
            new_cycle_factors_dict = {}

            for i1, factors_tpls1 in enumerate(a_c_factors, 0):
                for i2, factors_tpls2 in enumerate(a_c_factors[i1+1:], i1+1):
                    # if len(factors1) > 1 and len(factors2) > 1:
                    factors1 = factors_tpls1[-1]
                    factors2 = factors_tpls2[-1]

                    cycle1_tpl = factors_cycle_dict[factors_tpls1]
                    cycle2_tpl = factors_cycle_dict[factors_tpls2]

                    cycle1 = np.array(cycle1_tpl)
                    cycle2 = np.array(cycle2_tpl)

                    for a2, c2 in factors2:
                        cycle12 = (a2*cycle1+c2)%m
                    for a1, c1 in factors1:
                        cycle21 = (a1*cycle2+c1)%m

                    cycle12_tpl = tuple(cycle12)
                    cycle21_tpl = tuple(cycle21)

                    if not cycle12_tpl in cycle_factors_dict:
                        new_factors_tpl = cycle_factors_dict[cycle1_tpl]+(factors2, )
                        cycle_factors_dict[cycle12_tpl] = new_factors_tpl
                        factors_cycle_dict[new_factors_tpl] = cycle12_tpl
                        new_cycle_factors_dict[new_factors_tpl] = 0
                    if not cycle21_tpl in cycle_factors_dict:
                        new_factors_tpl = cycle_factors_dict[cycle2_tpl]+(factors1, )
                        cycle_factors_dict[cycle21_tpl] = new_factors_tpl
                        factors_cycle_dict[new_factors_tpl] = cycle21_tpl
                        new_cycle_factors_dict[new_factors_tpl] = 0

            print("len(new_cycle_factors_dict): {}".format(len(new_cycle_factors_dict)))
            print("len(cycle_factors_dict): {}".format(len(cycle_factors_dict)))

            if len(new_cycle_factors_dict) == 0:
                break

        return cycle_factors_dict

    # get_all_possible_cycles(full_cycles, m)

    # arr = np.zeros((m, m), dtype=np.int)
    # a = np.arange(0, m).reshape((-1, 1))
    # c = a.T

    # scale = 2
    # for x in range(0, m):
    #     arr = (a.dot(x)+c)%m
    #     img = get_img(arr, m=m)
    #     img = img.resize((img.width*scale, img.height*scale))
    #     ShowImg(img)

    # v = np.arange(0, m)
    # arr = np.sort((v.reshape((-1, 1)).dot(v.reshape((1, -1))).reshape((1, m, m))+v.reshape((1, 1, -1)))%m, axis=2).reshape((-1, m))
    # idxs = np.where(np.all((arr[:, 1:]-arr[:, :-1])==1, axis=1))[0]



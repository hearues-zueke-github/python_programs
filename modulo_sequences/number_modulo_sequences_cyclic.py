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
        # x_iter = vals
        x_iter = zeros
        # x_iter = ((a*x_iter)+vals)%m
        arr[:, 0] = x_iter
        for l in range(1, m, 1):
            x_iter = ((a*x_iter)+vals)%m
            arr[:, l] = x_iter
        for k, row in enumerate(arr, 0):
            t = (a, k)
            if np.unique(row).shape[0]==m:
                full_cycles[t] = row.tolist()
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


def get_full_cycles_2(m):
    found_full_cycles = {}

    max_cycle_len = 0
    for a in range(0, m):
        for b in range(0, m):
            for c in range(0, m):
                for x1 in range(0, m):
                    cycle = [0, x1]
                    cycle.append((a*cycle[-1]**2+b*cycle[-2]+c)%m)
                    cycle.append((a*cycle[-1]**2+b*cycle[-2]+c)%m)
                    cycle.append((a*cycle[-1]**2+b*cycle[-2]+c)%m)
                    cycle.append((a*cycle[-1]**2+b*cycle[-2]+c)%m)
                    cycle.append((a*cycle[-1]**2+b*cycle[-2]+c)%m)
                    cycle.append((a*cycle[-1]**2+b*cycle[-2]+c)%m)
                    cycle.append((a*cycle[-1]**2+b*cycle[-2]+c)%m)
                    cycle.append((a*cycle[-1]**2+b*cycle[-2]+c)%m)
                    cycle.append((a*cycle[-1]**2+b*cycle[-2]+c)%m)
                    cycle.append((a*cycle[-1]**2+b*cycle[-2]+c)%m)
                    # for _ in range(0, m**2):
                    cycle = cycle[-2:]
                    while True:
                        v = (a*cycle[-1]**2+b*cycle[-2]+c)%m
                        cycle.append(v)
                        if cycle[:2]==cycle[-2:]:
                            break
                    # unique_length = np.unique(cycle[2:]).shape[0]
                    # if  unique_length==m:
                    # if cycle[:2]==cycle[-2:]:
                    # arr = np.array(cycle)[:-1]
                    # arr2 = np.vstack((arr[:-1], arr[1:])).T*m**np.arange(1, -1, -1)
                    # sum_rows = np.sum(arr2, axis=1)
                    # print("arr2: {}".format(arr2))
                    # print("sum_rows: {}".format(sum_rows))
                    # if np.unique(sum_rows).shape[0]==m**2:
                    cycle = cycle[:-1]
                    key = (a,b,c,x1)
                    found_full_cycles[key] = cycle
                    if max_cycle_len < len(cycle):
                        max_cycle_len = len(cycle)
                        print("m: {}, cycle: {}".format(m, cycle))
                        print("m: {}, key: {}".format(m, key))

    max_cycle_len = np.max(list(map(lambda x: len(x), found_full_cycles.values())))
    # print("m: {}, max_cycle_len: {}".format(m, max_cycle_len))

    return max_cycle_len
    # return found_full_cycles


def get_one_full_cycle(m):
    complete_cycles = {}
    
    for a in range(0, m):
        for c in range(0, m):
            calc_v = lambda x1: (a*x1+c)%m

            mapping_dict = {}
            for x1 in range(0, m):
                mapping_dict[x1] = calc_v(x1)

            # get the full cycle lengths!
            for k in mapping_dict:
                cycle = [k]
                while True:
                    next_k = mapping_dict[cycle[-1]]
                    if next_k in cycle:
                        cycle.append(next_k)
                        break
                    cycle.append(next_k)
                if cycle[0]==cycle[-1] and len(cycle)>2:
                    complete_cycles[(a, c, k)] = cycle[:-1]

    return complete_cycles


def get_one_full_cycle_2(m):
    print("m: {}".format(m))
    complete_cycles = {}
    
    max_length = 0
    for a in range(0, m):
        for b in range(0, m):
            for c in range(0, m):
                t = (a, b, c)
                # calc_v = lambda x1, x2: (a*x1+b*x2+c)%m
                calc_v = lambda x1, x2: (a*x1+b*x2**2+c)%m
                # calc_v = lambda x1, x2: (a*x1**2+b*x2+c)%m

                mapping_dict = {}
                for x1 in range(0, m):
                    for x2 in range(0, m):
                        mapping_dict[(x1, x2)] = (x2, calc_v(x1, x2))

                # get the full cycle lengths!
                all_used_k = set()
                for k in mapping_dict:
                    if k in all_used_k:
                        continue

                    cycle = [k]
                    while True:
                        next_k = mapping_dict[cycle[-1]]
                        if next_k in all_used_k:
                            break
                        if next_k in cycle:
                            cycle.append(next_k)
                            break
                        cycle.append(next_k)
                    # all_used_k.update(set(cycle[:-1]))

                    if cycle[0]==cycle[-1] and len(cycle)>2:
                        # if t in complete_cycles:
                        #     break
                        cycle = cycle[:-1]
                        length = len(cycle)
                        if max_length < length:
                            max_length = length
                            # complete_cycles = {}
                        # if max_length == length:
                        complete_cycles[t+k] = cycle
                        all_used_k.update(set(cycle))
                        # complete_cycles[(a, b, c)+k] = cycle[:-1]

    return complete_cycles


def get_one_full_cycle_2_lst(nn):
    lst_len = []
    lst_max_len_amount = []
    lst_amount = []

    for n in range(2, nn+1):
        y = get_one_full_cycle_2(n)
        u, c = np.unique(list(map(lambda x: len(x), y.values())), return_counts=True)
        lst_len.append(u[-1])
        lst_max_len_amount.append(c[-1])
        lst_amount.append(u.shape[0])

    return lst_len, lst_max_len_amount, lst_amount

'''
calc_v = lambda x1, x2: (a*x1**2+b*x2+c)%m
In [244]: [len(list(get_one_full_cycle_2(n).values())[0]) for n in range(2, 23)]
Out[244]: [4, 4, 6, 8, 12, 12, 12, 16, 28, 25, 12, 39, 36, 28, 24, 43, 48, 46, 42, 36, 100]
In [253]: get_one_full_cycle_2_lst(22)
Out[253]: [1, 4, 8, 12, 14, 12, 32, 54, 4, 10, 200, 12, 18, 16, 256, 16, 108, 18, 32, 72, 10]

In [285]: get_one_full_cycle_2_lst(10)
Out[285]: ([4, 4, 6, 8, 12, 12, 12, 16, 28], [1, 4, 8, 12, 14, 12, 32, 54, 4])
In [288]: get_one_full_cycle_2_lst(12)
Out[288]: 
([4, 4, 6, 8, 12, 12, 12, 16, 28, 25, 12],
 [1, 4, 8, 12, 14, 12, 32, 54, 4, 10, 200],
 [3, 3, 4, 7, 5, 10, 6, 7, 15, 22, 5])

calc_v = lambda x1, x2: (a*x1+b*x2**2+c)%m
In [294]: get_one_full_cycle_2_lst(12)
Out[294]: 
([4, 8, 12, 20, 28, 37, 24, 42, 76, 120, 84],
 [1, 2, 8, 4, 2, 12, 64, 54, 8, 20, 16],
 [3, 7, 6, 17, 15, 33, 8, 17, 37, 74, 21])

calc_v = lambda x1, x2: (a*x1+b*x2+c)%m
In [226]: [len(list(get_one_full_cycle_2(n).values())[0]) for n in range(2, 16)]
Out[226]: [4, 8, 8, 24, 24, 48, 16, 24, 60, 120, 24, 168, 84, 60]
In [225]: get_one_full_cycle_2_lst(15)
Out[225]: [1, 6, 4, 20, 12, 56, 32, 144, 36, 176, 136, 312, 52, 180]

In [291]: get_one_full_cycle_2_lst(12)
Out[291]: 
([4, 8, 8, 24, 24, 48, 16, 24, 60, 120, 24],
 [1, 6, 8, 20, 12, 56, 128, 432, 36, 176, 656],
 [3, 5, 5, 10, 7, 13, 7, 9, 13, 19, 7])
'''

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

    # nn = int(sys.argv[1])

    # cycles = get_full_cycles_faster(5)

    used_cycles = {}
    unique_cycles = {}
    m = 8
    for a1 in range(0, m):
     for c1 in range(0, m):
      # for a2 in range(0, m):
      #  for c2 in range(0, m):
      #   calc_v = lambda x: (a2*((a1*x+c1)%m)+c2)%m
        calc_v = lambda x: (a1*x+c1)%m
        mapping_dict = {i: calc_v(i) for i in range(0, m)}
        # all_used_v = set()
        cycle = [calc_v(0)]
        while True:
            v = calc_v(cycle[-1])
            if v in cycle:
                cycle.append(v)
                break
            cycle.append(v)
        # if cycle[0]==cycle[-1]:
        if True:
            cycle = cycle[cycle.index(cycle[-1]):]
            t = (a1, c1)
            # t = (a1, c1, a2, c2)
            cycle = cycle[:-1]
            used_cycles[t] = cycle
            tc = tuple(cycle)
            if not tc in unique_cycles:
                unique_cycles[tc] = t
    print("used_cycles: {}".format(used_cycles))
    print("unique_cycles: {}".format(unique_cycles))


    sys.exit(0)

    lst_max_cycles = []
    lst_amounts_a_c = []
    lst_amounts_a_bigger_1 = []

    for m in range(1, nn+1):
        arr = np.array(list(get_full_cycles_faster(m).keys())).T
        lst_max_cycles.append(arr.shape[1])
        amount_a = np.unique(arr[0]).shape[0]
        amount_c = np.unique(arr[1]).shape[0]
        lst_amounts_a_c.append((m, amount_a, amount_c))
        if amount_a > 1:
            lst_amounts_a_bigger_1.append((m, amount_a, amount_c))

    print("lst_max_cycles: {}".format(lst_max_cycles))
    # print("lst_amounts_a_c: {}".format(lst_amounts_a_c))
    # print("lst_amounts_a_bigger_1: {}".format(lst_amounts_a_bigger_1))

    lst_amounts_c = reduce(lambda a, b: a+[b[2]], lst_amounts_a_c, [])
    print("lst_amounts_c: {}".format(lst_amounts_c))

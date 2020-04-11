import os
import sys

from os.path import expanduser
PATH_HOME = expanduser("~")+'/'

from PIL import Image

import numpy as np

from PIL import ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style

from copy import deepcopy

sys.path.append("../")
import global_object_getter_setter

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


def get_arr(used_length=5000000):
    file_path_enwik8 = PATH_HOME+'Downloads/enwik8'

    print("file_path_enwik8: {}".format(file_path_enwik8))

    if not os.path.exists(file_path_enwik8):
        print("Please make sure, that '{}' does exist!".format(file_path_enwik8))
        sys.exit(-1)

    with open(file_path_enwik8, 'rb') as f:
        data = f.read()

    if used_length==-1:
        arr = np.array(list(data), dtype=np.uint8)
    elif used_length>=0:
        arr = np.array(list(data[:used_length]), dtype=np.uint8)
    else:
        assert False=='used_length<0 !'
    # arr = np.array(list(data[:used_length]), dtype=object)
    # global_object_getter_setter.save_object('data', data)

    print("arr.shape: {}".format(arr.shape))

    return arr


def check_merged_idxs_ranges(idxs_ranges):
    fails = 0
    for i, (t1, t2) in enumerate(zip(idxs_ranges[:-1], idxs_ranges[1:]), 0):
        try:
            assert t1[1]<=t2[0]
        except:
            print("i: {}".format(i))
            print("t1: {}, t2: {}".format(t1, t2))
            fails += 1
    if fails > 0:
        sys.exit("FAIL!!")


def do_merge_idxs_ranges(idxs_ranges_1, idxs_ranges_2, is_inplace=False):
    # assume: no overlap in each idxs_ranges list is given!
    # assume: idxs_ranges is already sorted!
    # e.g.: idxs_ranges_1 = ir1 = [(2, 3), (5, 7), (10, 13)]
    # e.g.: idxs_ranges_2 = ir2 = [(3, 5), (6, 9), (11, 12)]

    if len(idxs_ranges_1)==0:
        if is_inplace:
            idxs_ranges_1.extend(idxs_ranges_2)
            return idxs_ranges_1
        return deepcopy(idxs_ranges_2)

    if is_inplace:
        idxs_ranges_merged = idxs_ranges_1
    else:
        idxs_ranges_merged = deepcopy(idxs_ranges_1)

    len1 = len(idxs_ranges_1)
    len2 = len(idxs_ranges_2)
    i = 0
    j = 0
    is_prev_valid = True
    while i < len1 and j < len2:
        x1, x2 = idxs_ranges_1[i]
        y1, y2 = idxs_ranges_2[j]

        if (x2 <= y1):
            i += 1
            is_prev_valid = True
        elif (y2 <= x1):
            j += 1
            if is_prev_valid:
                idxs_ranges_merged.append((y1, y2))
            is_prev_valid = True
        elif (x1 == y1 and x2 == y2):
            i += 1
            j += 1
            is_prev_valid = True
        elif (x1 == y1 and x2 < y2):
            i += 1
            is_prev_valid = False
        elif (x1 == y1 and y2 < x2):
            j += 1
            is_prev_valid = False
        elif (x2 == y2):
            i += 1
            j += 1
            is_prev_valid = True
        elif (x1 < y1 and y2 < x2):
            j += 1
            is_prev_valid = False
        elif (y1 < x1 and x2 < y2):
            i += 1
            is_prev_valid = False
        elif (x1 < y1 and y1 < x2 and x2 < y2):
            i += 1
            is_prev_valid = False
        elif (y1 < x1 and x1 < y2 and y2 < x2):
            j += 1
            is_prev_valid = True
        else:
            assert False
        # TODO: find the error(s)!!!

    # if i==len1 and not is_prev_valid: #((x1 < y1 and y1 < x2 and x2 < y2) or (x1 == y1)):
    if i==len1 and ((x1 < y1 and y1 < x2 and x2 < y2) or (x1 == y1) or (y1 < x1 and x2 < y2)):
        j += 1

    for t in idxs_ranges_2[j:]:
        idxs_ranges_merged.append(t)

    return sorted(idxs_ranges_merged)


def get_list_of_ranges(idxs_ranges):
    assert len(idxs_ranges.shape)==2
    assert idxs_ranges.shape[1]==2
    diff = idxs_ranges[0, 1]-idxs_ranges[0, 0]
    assert np.all(np.diff(idxs_ranges, axis=1)==diff)
    assert np.all(np.diff(idxs_ranges[:, 0])>0)

    # print("idxs_ranges:\n{}".format(idxs_ranges))
    global_object_getter_setter.save_object('idxs_ranges', idxs_ranges)

    idxs = idxs_ranges[1:, 0]<idxs_ranges[:-1, 1]
    idxs2 = np.hstack(((False, ), idxs))|np.hstack((idxs, (False, )))

    idxs_ranges_out_range = idxs_ranges[idxs2]
    idxs_ranges_ok = idxs_ranges[~idxs2]
    # print("idxs_ranges_out_range:\n{}".format(idxs_ranges_out_range))
    # print("idxs_ranges_ok:\n{}".format(idxs_ranges_ok))

    a = idxs_ranges_out_range[:, 0]
    idxs_parts = np.hstack(((0, ), np.where(np.diff(a)>=2)[0]+1, (a.shape[0], )))
    # print("idxs_parts:\n{}".format(idxs_parts))

    # TODO: 2020.04.10: can be made, such that all possible combinations are made!
    l_idxs_ranges_parts = [[np.arange(i1+j, i2, diff) for j in range(0, diff)] for i1, i2 in np.vstack((idxs_parts[:-1], idxs_parts[1:])).T]
    # print("l_idxs_ranges_parts: {}".format(l_idxs_ranges_parts))

    l_combined_ranges = []
    for r in zip(*l_idxs_ranges_parts):
        l_combined_ranges.append(np.hstack(r))
    # print("l_combined_ranges: {}".format(l_combined_ranges))

    l_idxs_ranges = [(lambda x: x[np.argsort(x[:, 0])])(np.vstack((idxs_ranges_out_range[idxs], idxs_ranges_ok))) for idxs in l_combined_ranges]
    # print("l_idxs_ranges: {}".format(l_idxs_ranges))

    return l_idxs_ranges


def check_if_crossover_is_possible(arr1, arr2):
    len1 = arr1.shape[0]
    len2 = arr2.shape[0]

    if len1==len2:
        for i in range(1, len1):
            if np.all(arr1[i:]==arr2[:-i]):
                return True

        if np.all(arr1==arr2):
            return True

        for i in range(1, len1):
            if np.all(arr1[:-i]==arr2[i:]):
                return True
    else:
        if len1<len2:
            len1, len2 = len2, len1
            arr1, arr2 = arr2, arr1

        for i in range(1, len2):
            if np.all(arr1[:i]==arr2[-i:]):
                return True

        for i in range(0, len1-len2+1):
            if np.all(arr1[i:i+len2]==arr2):
                return True

        for i in range(1, len2):
            if np.all(arr1[-i:]==arr2[:i]):
                return True

    return False


def do_some_simple_tests():
    idxs_ranges_1 = [(1, 2), (4, 6)]
    idxs_ranges_2 = [(2, 4)]
    idxs_ranges_merged_1 = do_merge_idxs_ranges(idxs_ranges_1, idxs_ranges_2)
    assert idxs_ranges_merged_1==[(1, 2), (2, 4), (4, 6)]

    idxs_ranges_1 = [(1, 2), (4, 6), (9, 11)]
    idxs_ranges_2 = [(2, 4), (4, 5), (6, 9)]
    idxs_ranges_merged_2 = do_merge_idxs_ranges(idxs_ranges_1, idxs_ranges_2)
    assert idxs_ranges_merged_2==[(1, 2), (2, 4), (4, 6), (6, 9), (9, 11)]

    idxs_ranges_1 = []
    idxs_ranges_2 = [(2, 4), (4, 5), (6, 9)]
    idxs_ranges_merged_3 = do_merge_idxs_ranges(idxs_ranges_1, idxs_ranges_2)
    assert idxs_ranges_merged_3==[(2, 4), (4, 5), (6, 9)]

    idxs_ranges_1 = [(1, 2)]
    idxs_ranges_2 = [(2, 4), (4, 5), (6, 9)]
    idxs_ranges_merged_4 = do_merge_idxs_ranges(idxs_ranges_1, idxs_ranges_2)
    # print("idxs_ranges_merged_4: {}".format(idxs_ranges_merged_4))
    assert idxs_ranges_merged_4==[(1, 2), (2, 4), (4, 5), (6, 9)]

    idxs_ranges_1 = [(1, 2), (4, 6)]
    idxs_ranges_2 = [(2, 4), (4, 5), (6, 9)]
    idxs_ranges_merged_5 = do_merge_idxs_ranges(idxs_ranges_1, idxs_ranges_2)
    assert idxs_ranges_merged_5==[(1, 2), (2, 4), (4, 6), (6, 9)]
    
    idxs_ranges_1 = [(0, 4)]
    idxs_ranges_2 = [(2, 5)]
    idxs_ranges_merged_6 = do_merge_idxs_ranges(idxs_ranges_1, idxs_ranges_2)
    # print("idxs_ranges_merged_6: {}".format(idxs_ranges_merged_6))
    assert idxs_ranges_merged_6==[(0, 4)]

    idxs_ranges_1 = [(0, 2), (3, 7)]
    idxs_ranges_2 = [(2, 3), (5, 9)]
    idxs_ranges_merged_7 = do_merge_idxs_ranges(idxs_ranges_1, idxs_ranges_2)
    assert idxs_ranges_merged_7==[(0, 2), (2, 3), (3, 7)]

    idxs_ranges_1 = [(5, 9), (13, 18), (21, 25)]
    idxs_ranges_2 = [(7, 11), (12, 14), (16, 19), (20, 25)]
    idxs_ranges_merged_8 = do_merge_idxs_ranges(idxs_ranges_1, idxs_ranges_2)
    # print("idxs_ranges_merged_8: {}".format(idxs_ranges_merged_8))
    # assert idxs_ranges_merged_8==[(0, 2), (7, 9), (10, 12)]

    lst_idxs_ranges = [
        idxs_ranges_merged_1,
        idxs_ranges_merged_2,
        idxs_ranges_merged_3,
        idxs_ranges_merged_4,
        idxs_ranges_merged_5,
        idxs_ranges_merged_6,
        idxs_ranges_merged_7,
        idxs_ranges_merged_8,
    ]
    for idxs_ranges in lst_idxs_ranges:
        # print("idxs_ranges: {}".format(idxs_ranges))
        if len(idxs_ranges) < 2:
            continue
        for t1, t2 in zip(idxs_ranges[:-1], idxs_ranges[1:]):
            assert t1[1]<=t2[0]


    l = get_list_of_ranges(np.array([
        [1, 3],
        [2, 4],
        [3, 5],
        [8, 10],
        [14, 16],
        [15, 17],
        [16, 18],
        [20, 22],
        [21, 23],
        [26, 28],
    ]))

    l = get_list_of_ranges(np.array([
        [1, 4],
        [2, 5],
        [3, 6],
        [8, 11],
        [15, 18],
        [17, 20],
        [20, 23],
        [21, 24],
        [26, 29],
    ]))
    print("l: {}".format(l))


do_some_simple_tests()


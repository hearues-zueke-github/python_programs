import os

from os.path import expanduser
PATH_HOME = expanduser("~")+'/'

from PIL import Image

import numpy as np

from PIL import ImageTk
from tkinter import Tk, Label, BOTH
from tkinter.ttk import Frame, Style

from copy import deepcopy

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


def get_arr():
    file_path_enwik8 = PATH_HOME+'Downloads/enwik8'

    print("file_path_enwik8: {}".format(file_path_enwik8))

    if not os.path.exists(file_path_enwik8):
        print("Please make sure, that '{}' does exist!".format(file_path_enwik8))
        sys.exit(-1)

    with open(file_path_enwik8, 'rb') as f:
        data = f.read()

    used_length = 5000000
    # used_length = 3000000
    data_str = data.decode('utf-8')[:used_length]
    lst_data = list(data)[:used_length]
    arr = np.array(lst_data, dtype=object)

    print("used_length: {}".format(used_length))

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
    print("idxs_ranges_merged_8: {}".format(idxs_ranges_merged_8))
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
        print("idxs_ranges: {}".format(idxs_ranges))
        if len(idxs_ranges) < 2:
            continue
        for t1, t2 in zip(idxs_ranges[:-1], idxs_ranges[1:]):
            assert t1[1]<=t2[0]

do_some_simple_tests()

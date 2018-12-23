#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import dill
import functools
import sys
import time

import numpy as np

from PIL import Image, ImageTk

import multiprocessing as mp
from multiprocessing import Process, Pipe, Lock

def get_arr_shifts(arr, length):
    arr_length = arr.shape[0]
    arr_shifts = np.zeros((length, arr_length+length-1), dtype=np.uint8)

    for i in range(0, length):
        arr_shifts[i, i:arr_length+i] = arr
 
    arr_shifts = arr_shifts.T
    arr_shifts = arr_shifts[length-1:-length]

    return arr_shifts


def find_max_match(arr):
    previous_pattern = None
    length = arr.shape[0]
    for i in range(1, length//2+1):
        if np.all(arr[:i]==arr[i:2*i]):
            if isinstance(previous_pattern, type(None)):
                previous_pattern = arr[:i]
            else:
                pattern = arr[:i]
                # Test if this previous pattern is really appliable on the new pattern or not!
                length_prev = previous_pattern.shape[0]
                length_now = pattern.shape[0]

                if (length_prev == 1) or \
                   (length_now % length_prev > 0) or \
                   (not np.all(pattern.reshape((-1, length_prev))==previous_pattern)):
                    previous_pattern = pattern

    return previous_pattern


def find_first_repeat_pattern(arr):
    arr_reverse = arr[::-1]

    pattern = find_max_match(arr_reverse)

    if isinstance(pattern, type(None)):
        return None, None

    length_arr = arr.shape[0]
    length_pattern = pattern.shape[0]

    last_possible_i = 0
    for i in range(1, length_arr//length_pattern+1):
        if np.all(arr_reverse[length_pattern*i:length_pattern*(i+1)]==pattern):
            last_possible_i = i
        else:
            break

    # pattern = arr_reverse[length_pattern*last_possible_i:length_pattern*(last_possible_i+1)]
    rest_arr = arr_reverse[length_pattern*(last_possible_i+1):]

    # print("last_possible_i: {}".format(last_possible_i))
    # print("i: {}".format(i))

    # print("pattern:\n{}".format(pattern))
    # print("rest_arr:\n{}".format(rest_arr))

    same_length = np.min((pattern.shape[0], rest_arr.shape[0]))
    equal_numbers = pattern[:same_length]==rest_arr[:same_length]
    # print("equal_numbers:\n{}".format(equal_numbers))

    eql_nums_sum = np.cumsum(equal_numbers)
    rest_nums = eql_nums_sum[~equal_numbers]

    # print("eql_nums_sum:\n{}".format(eql_nums_sum))
    # print("rest_nums:\n{}".format(rest_nums))

    last_idx = np.min(rest_nums)
    # print("pattern[:last_idx]:\n{}".format(pattern[:last_idx]))
    # print("rest_arr[:last_idx]:\n{}".format(rest_arr[:last_idx]))

    idx1 = length_pattern*last_possible_i+last_idx
    idx2 = length_pattern*(last_possible_i+1)+last_idx
    first_pattern = arr_reverse[idx1:idx2][::-1]
    # print("first_pattern:\n{}".format(first_pattern))

    return first_pattern, length_arr-idx2


if __name__ == "__main__":
    lst = [1] # starting sequence
    n = 1000-1
    m = 10 # modulo
    for i in range(len(lst)-1, n):
        s = 0
        j = i
        acc = 0
        multiplier = 1
        while j >= 0:
            x = lst[j]
            s = (s+x*multiplier) % m
            multiplier = (multiplier+1) % m
            acc += x+1
            j -= acc
        lst.append(s % m)

    print("lst:\n{}".format(lst))
    arr = np.array(lst)

    with open("b_sequence.txt", "w") as fout:
        for i, x in enumerate(arr, 1):
            fout.write("{} {}\n".format(i, x))

    if False:
        arr_pattern, idx = find_first_repeat_pattern(arr)

        if idx == None:
            print("No Pattern could be found! More information needed!")
        else:
            pattern_length = arr_pattern.shape[0]

            first_part = arr[:idx]
            amount_patterns = (arr.shape[0]-idx)//pattern_length
            last_part = arr[idx+amount_patterns*pattern_length:]

            print("\nfirst_part.shape[0]: {}".format(first_part.shape[0]))
            print("first_part:\n{}".format(first_part))
            
            print("\namount_patterns: {}".format(amount_patterns))
            print("arr_pattern.shape[0]: {}".format(arr_pattern.shape[0]))
            print("arr_pattern:\n{}".format(arr_pattern))
            
            print("\nlast_part.shape[0]: {}".format(last_part.shape[0]))
            print("last_part:\n{}".format(last_part))

    if False:
        s = arr[0]
        arr2 = [s]
        length = arr.shape[0]
        for i in range(1, length):
            s1 = s
            is_found = False
            for a in arr[i:]:
                s1 -= a+1
                if s1 < 1:
                    break
                if not s1 in arr2:
                    arr2.append(s1)
                    is_found = True
                    break

            if not is_found:
                s += arr[i]+1
                arr2.append(s)

            # print("is_found: {}, s: {}, s1: {}".format(is_found, s, s1))
            # input()
        print("arr2:\n{}".format(arr2))

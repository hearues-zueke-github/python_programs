#! /usr/bin/python2.7

import sys

import numpy as np

from PIL import Image

def get_valid_binary_row(n):
    row = np.zeros((n, )).astype(np.int)

    idx = 0
    color = np.random.randint(0, 2)
    length = np.random.randint(1, 3)
    while idx < n:
        row[idx:idx+length] = color
        idx += length

        color = (color+1)%2
        length = np.random.randint(1, 3)

    return row

if __name__ == "__main__":
    n = 15
    # arr = np.random.randint(0, 2, (n, n))

    # arr_dy = (arr[1:]==arr[:-1])+0
    # arr_dx = (arr[:, 1:]==arr[:, :-1])+0

    # print("arr:\n{}".format(arr))

    # print("arr_dy:\n{}".format(arr_dy))
    # print("arr_dx:\n{}".format(arr_dx))

    # arr_sum_dy_2 = arr_dy[:-1]+arr_dy[1:]
    # arr_sum_dx_2 = arr_dx[:, :-1]+arr_dx[:, 1:]

    # print("arr_sum_dy_2:\n{}".format(arr_sum_dy_2))
    # print("arr_sum_dx_2:\n{}".format(arr_sum_dx_2))


    # m = 20
    # resize_factor = 10
    # arr = np.zeros((m, n)).astype(np.int)
    # for i in xrange(0, m):
    #     arr[i] = get_valid_binary_row(n)

    # new_arr = np.zeros((resize_factor**2, m*n)).astype(np.uint8)
    # new_arr[:] = arr.reshape((-1, ))
    # new_arr = new_arr.T.reshape((m, n*resize_factor, resize_factor)) \
    #                    .transpose(0, 2, 1) \
    #                    .reshape((m*resize_factor, n*resize_factor))

    # # print("new_arr:\n{}".format(new_arr.tolist()))

    # # img = Image.fromarray(arr.astype(np.uint8)*255)
    # img = Image.fromarray(new_arr.astype(np.uint8)*255)
    # # img = img.resize((n*3, n*3), Image.ANTIALIAS)
    # img.show()

    # sys.exit(0)


    def set_rows(arr, n, amount):
        for i in xrange(0, amount):
            arr[i] = get_valid_binary_row(n)

    def get_amount_same_numbers(arr):
        sum_rows = 0
        rows = arr.shape[0]
        # print("arr.shape[0]: {}".format(arr.shape[0]))
        # sys.exit(0)
        row_0 = arr[0]
        for i in xrange(1, rows):
            sum_rows += row_0==arr[i]

        print("sum_rows:\n{}".format(sum_rows))

        return sum_rows

    arr = np.zeros((n, n)).astype(np.int)
    same_numbers = 2
    # arr[0] = get_valid_binary_row(n)
    # arr[1] = get_valid_binary_row(n)
    set_rows(arr, n, same_numbers)

    i = same_numbers
    worked_idx = [2]
    tries = 0
    while i < n:
        if i < 2:
            arr[i] = get_valid_binary_row(n)
            i += 1
            print("i: {}".format(i))
            worked_idx.append(i)
        arr[i] = get_valid_binary_row(n)
        # print("i: {}, arr[{}]: {}".format(i, i, arr[i]))
        if np.where(get_amount_same_numbers(arr[i-same_numbers:i+1])>=same_numbers)[0].shape[0] > 0:
        # if np.where((0+(arr[i-2]==arr[i-1])+(arr[i-2]==arr[i]))>=same_numbers)[0].shape[0] > 0:
            tries += 1
            if tries > 100:
                tries = 0
                i -= 1
                print("i: {}".format(i))
                worked_idx.append(i)
            continue
        tries = 0
        i += 1
        print("i: {}".format(i))
        worked_idx.append(i)

    print("arr:\n{}".format(arr))
    print("worked_idx:\n{}".format(worked_idx))

    resize_factor = 10

    new_arr = np.zeros((resize_factor**2, n*n)).astype(np.uint8)
    new_arr[:] = arr.reshape((-1, ))
    new_arr = new_arr.T.reshape((n, n*resize_factor, resize_factor)) \
                       .transpose(0, 2, 1) \
                       .reshape((n*resize_factor, n*resize_factor))

    # print("new_arr:\n{}".format(new_arr.tolist()))

    # img = Image.fromarray(arr.astype(np.uint8)*255)
    img = Image.fromarray(new_arr.astype(np.uint8)*255)
    # img = img.resize((n*3, n*3), Image.ANTIALIAS)
    img.show()

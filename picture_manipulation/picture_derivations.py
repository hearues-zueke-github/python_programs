#! /usr/bin/python3.5

import numpy as np

from PIL import Image

def padding_one_pixel(pix):
    shape = pix
    pix_1 = np.vstack((pix[0], pix, pix[-1])).T
    pix_2 = np.vstack((pix_1[0], pix_1, pix_1[-1])).T
    return pix_2

shape = (5, 6)
pix = np.random.randint(0, 10, shape)

print("pix:\n{}".format(pix))

part = pix[0:3, 0:3]
print("part:\n{}".format(part))

pix_pad = padding_one_pixel(pix)
print("pix_pad:\n{}".format(pix_pad))

idx_y, idx_x = np.array([(j, i) for j in range(0, 3) for i in range(0, 3)]).T

idx_y_row = np.zeros((0, 9)).astype(np.int)
idx_x_row = np.zeros((0, 9)).astype(np.int)
for j in range(0, shape[0]):
    for i in range(0, shape[1]):
        idx_y_row = np.vstack((idx_y_row, idx_y+j))
        idx_x_row = np.vstack((idx_x_row, idx_x+i))

idx_y_row = idx_y_row.reshape((-1, ))
idx_x_row = idx_x_row.reshape((-1, ))

print("idx_y: {}".format(idx_y))
print("idx_x: {}".format(idx_x))
print("idx_y_row:\n{}".format(idx_y_row))
print("idx_x_row:\n{}".format(idx_x_row))
print("pix_pad[idx_y, idx_x]: {}".format(pix_pad[idx_y, idx_x]))
print("pix_pad[idx_y_row, idx_x_row]:\n{}".format(pix_pad[idx_y_row, idx_x_row]))
pix_deriv_table = pix_pad[idx_y_row, idx_x_row].reshape(shape[0]*shape[1], 3*3)
print("pix_deriv_table:\n{}".format(pix_deriv_table))

mask_sobel = np.zeros((3, 3)).astype(np.int)
mask_sobel[0] = -1
mask_sobel[2] = 1
mask_x = mask_sobel.copy().reshape((-1, ))
mask_y = mask_sobel.T.copy().reshape((-1, ))

print("mask_x: {}".format(mask_x))
print("mask_y: {}".format(mask_y))

pix_deriv_x = np.sum(pix_deriv_table*mask_x, axis=1).reshape(shape)
pix_deriv_y = np.sum(pix_deriv_table*mask_y, axis=1).reshape(shape)

print("pix_deriv_x: {}".format(pix_deriv_x))
print("pix_deriv_y: {}".format(pix_deriv_y))

if __name__ == "__main__":
    print("Hello World!")

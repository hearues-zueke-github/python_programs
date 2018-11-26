#! /usr/bin/python3.6

import os

import numpy as np

from PIL import Image

if __name__ == "__main__":
    def show_image(pix):
        Image.fromarray(pix).show()
    
    img = Image.open("images/fall-autumn-red-season_resized.jpg")
    pix = np.array(img)

    pix_r, pix_g, pix_b = pix[:, :, 0], pix[:, :, 1], pix[:, :, 2]

    calc_x_diff = lambda pix: (lambda x: x[:, 1:]-x[:, :-1])(np.hstack((pix[:, 0].reshape((-1, 1)), pix)).astype(np.int8))
    calc_y_diff = lambda pix: (lambda x: x[1:]-x[:-1])(np.vstack((pix[0], pix)).astype(np.int8))
    pix_r_x_diff = calc_x_diff(pix_r)
    pix_r_y_diff = calc_y_diff(pix_r)

    threshold = 40
    pix_r_x_diff_th = pix_r_x_diff.copy()
    used_pixs = np.abs(pix_r_x_diff) < threshold
    pix_r_x_diff_th[used_pixs] = 0
    pix_r_x_diff_th[~used_pixs] = 255

    pix_r_y_diff_th = pix_r_y_diff.copy()
    used_pixs = np.abs(pix_r_y_diff) < threshold
    pix_r_y_diff_th[used_pixs] = 0
    pix_r_y_diff_th[~used_pixs] = 255

    pix_r_xx_diff = calc_x_diff(pix_r_x_diff)
    pix_r_xx_diff_th = pix_r_xx_diff.copy()
    used_pixs = np.abs(pix_r_xx_diff) < threshold
    pix_r_x_diff_th[used_pixs] = 0
    pix_r_x_diff_th[~used_pixs] = 255

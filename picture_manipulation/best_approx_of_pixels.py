#! /usr/bin/python3.5

import os

import numpy as np

from PIL import Image

def create_new_color_palette(amount):
    Image.fromarray(np.random.randint(0, 256, (1, amount, 3)).astype(np.uint8)).save("color_palette_2.png", "PNG")

def sort_by_row(pix):
    return pix[np.argsort(np.dot(pix, 256**np.arange(pix.shape[1]-1, -1, -1)))]

if __name__ == "__main__":
    # img = Image.open("nature_1.jpg").convert('LA')
    # img_idx = 3

    from os.path import expanduser
    home = expanduser("~")
    print("home: {}".format(home))
    
    path_images = home+"/Pictures/best_fits/"
    if not os.path.exists(path_images):
        os.makedirs(path_images)

    for img_idx in range(1, 4):
        img = Image.open("nature_{}.jpg".format(img_idx))
        pix_orig = np.array(img).astype(np.int)
        # pix_orig_color = np.array(img).astype(np.int)
        # pix_orig = np.dot(pix_orig_color.astype(np.float)[..., :3], [0.299, 0.587, 0.114]).astype(np.int)
            
        print("pix_orig.shape: {}".format(pix_orig.shape))
        # print("pix_orig_color.shape: {}".format(pix_orig_color.shape))
        # print("unique_used_pixels:\n{}".format(unique_used_pixels))
        # print("unique_used_pixels.shape: {}".format(unique_used_pixels.shape))

        # # Use some pixels from the picture by itself!
        unique_used_pixels = np.unique(pix_orig.reshape((-1, pix_orig.shape[-1])), axis=0)
        
        needed_idx = 50
   
        pix_palette_full = sort_by_row(unique_used_pixels.astype(np.int)).astype(np.uint8)
        rows_palette = pix_palette_full.shape[0]
        pix_palette = unique_used_pixels[np.arange(0, rows_palette, rows_palette//needed_idx)[:needed_idx]]

        # pix_palette = np.random.randint(0, 256, (needed_idx, 3)).astype(np.uint8)
        # pix_palette = np.array([[0x00, 0x00, 0x00],
        #                         [0xFF, 0x00, 0x00],
        #                         [0x00, 0xFF, 0x00],
        #                         [0x00, 0x00, 0xFF],
        #                         [0xFF, 0xFF, 0x00],
        #                         [0xFF, 0x00, 0xFF],
        #                         [0x00, 0xFF, 0xFF],
        #                         [0xFF, 0xFF, 0xFF]]).astype(np.uint8)
        
        img_palette = Image.fromarray(pix_palette.reshape((1, -1, pix_palette.shape[-1])).astype(np.uint8))
        img_palette.save(path_images+"color_palette_{}.png".format(img_idx), "PNG")

        # Load the pix_palette from anothe file!
        # pix_palette = np.array(Image.open("color_palette_1.png")) \
        # create_new_color_palette(100)
        # pix_palette = np.array(Image.open("color_palette_2.png")) \
        #                 .astype(np.int)
        # pix_palette = pix_palette.reshape((-1, pix_palette.shape[-1]))
        print("pix_palette.shape: {}".format(pix_palette.shape))
        # Image.fromarray(pix_palette).show()

        # TODO: split this part in e.g. 2x2 or 4x4 parts, to make it not so RAM intensive!
        diffs = np.abs(pix_orig.reshape((pix_orig.shape[0], pix_orig.shape[1], 1, pix_orig.shape[2]))-pix_palette)
        sums = np.dot(diffs, 256**np.arange(pix_orig.shape[-1]-1, -1, -1))
        best_idx = np.argmin(sums, axis=-1)
        # best_idx = np.argmin(np.abs(pix_orig.reshape(pix_orig.shape+(1, ))-pix_palette), axis=-1)

        pix_best_fit = pix_palette[best_idx]

        img_orig = Image.fromarray(pix_orig.astype(np.uint8))
        img_best_fit = Image.fromarray(pix_best_fit.astype(np.uint8))

        img_orig.save(path_images+"best_approx_img_nr_{}_orig.png".format(img_idx), "PNG")
        img_best_fit.save(path_images+"best_approx_img_nr_{}_best_fit.png".format(img_idx), "PNG")

        print("Finished picture #{}".format(img_idx))

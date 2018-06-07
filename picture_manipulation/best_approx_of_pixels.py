#! /usr/bin/python3.5

import numpy as np

from PIL import Image

def create_new_color_palette(amount):
    Image.fromarray(np.random.randint(0, 256, (1, amount, 3)).astype(np.uint8)).save("color_palette_2.png", "PNG")

if __name__ == "__main__":
    # img = Image.open("nature_1.jpg").convert('LA')
    img = Image.open("nature_1.jpg")
    pix_orig = np.array(img).astype(np.int)
    # pix_orig_color = np.array(img).astype(np.int)
    # pix_orig = np.dot(pix_orig_color.astype(np.float)[..., :3], [0.299, 0.587, 0.114]).astype(np.int)
        
    print("pix_orig.shape: {}".format(pix_orig.shape))
    # print("pix_orig_color.shape: {}".format(pix_orig_color.shape))
    # print("unique_used_pixels:\n{}".format(unique_used_pixels))
    # print("unique_used_pixels.shape: {}".format(unique_used_pixels.shape))

    # # Use some pixels from the picture by itself!
    unique_used_pixels = np.unique(pix_orig.reshape((-1, pix_orig.shape[-1])), axis=0)
    used_idx = np.random.permutation(np.arange(0, unique_used_pixels.shape[0]))[:30]
    pix_palette = unique_used_pixels[used_idx]
    Image.fromarray(pix_palette.reshape((1, -1, pix_palette.shape[-1])).astype(np.uint8)).save("color_palette_2.png", "PNG")

    # Load the pix_palette from anothe file!
    # pix_palette = np.array(Image.open("color_palette_1.png")) \
    # create_new_color_palette(100)
    # pix_palette = np.array(Image.open("color_palette_2.png")) \
    #                 .astype(np.int)
    # pix_palette = pix_palette.reshape((-1, pix_palette.shape[-1]))
    print("pix_palette.shape: {}".format(pix_palette.shape))
    # Image.fromarray(pix_palette).show()

    diffs = np.abs(pix_orig.reshape((pix_orig.shape[0], pix_orig.shape[1], 1, pix_orig.shape[2]))-pix_palette)
    sums = np.dot(diffs, 256**np.arange(pix_orig.shape[-1]-1, -1, -1))
    best_idx = np.argmin(sums, axis=-1)
    # best_idx = np.argmin(np.abs(pix_orig.reshape(pix_orig.shape+(1, ))-pix_palette), axis=-1)

    pix_best_fit = pix_palette[best_idx]

    img_orig = Image.fromarray(pix_orig.astype(np.uint8))
    img_best_fit = Image.fromarray(pix_best_fit.astype(np.uint8))

    img_orig.save("best_approx_img_orig.png", "PNG")
    img_best_fit.save("best_approx_img_best_fit.png", "PNG")

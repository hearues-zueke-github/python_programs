#! /usr/bin/python3.6

import os

import numpy as np

from PIL import Image

def get_pix_between(pix_1, pix_2, alpha=0.5):
    return (pix_1.astype(np.float)*alpha+pix_2.astype(np.float)*(1.-alpha)).astype(np.uint8)
    
def get_random_image(height, width):
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

if __name__ == "__main__":
    height = 256
    width = height

    n = 10
    pixs = [get_random_image(height, width) for _ in range(0, n)]
    arr_pix_combines = np.array(pixs)
    path_images = "images/random_sequence/"

    if not os.path.exists(path_images):
        os.makedirs(path_images)

    path_template = path_images+"rnd_{}_{}_i_{{:03}}_j_{{:02}}.png".format(height, width)
    for i, (pix_1, pix_2) in enumerate(zip(pixs[:-1], pixs[1:])):
        Image.fromarray(pix_1).save(path_template.format(i, 0))
        amount_combines = 10
        for j in range(1, amount_combines):
            print("i: {}, j: {}".format(i, j))
            Image.fromarray(get_pix_between(pix_1, pix_2, float(amount_combines-j)/amount_combines)).save(path_template.format(i, j))

    Image.fromarray(arr_pix_combines[-1]).save(path_template.format(arr_pix_combines.shape[0]-1, 0))

#! /usr/bin/python3.5

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

def get_pythagorian_triple(n):
    triple_numbers = []

    for x in range(1, n-1):
        for y in range(x+1, n):
            if x**2+y**2 in square_numbers:
                triple_numbers.append((x, y, square_roots[x**2+y**2]))

    return triple_numbers

def get_pythagorian_3_square_sums(n):
    sum_3_squares = []

    for x in range(1, n):
        for y in range(x, n):
            for z in range(y, n):
                if x**2+y**2+z**2 in square_numbers:
                    sum_3_squares.append((x, y, z, square_roots[x**2+y**2+z**2]))

    return sum_3_squares

n = 1000
square_numbers = [i**2 for i in range(1, 2*n)]
square_roots = {i**2: i for i in range(1, 2*n)}

sum_2_squares = get_pythagorian_triple(n)
sum_3_squares = get_pythagorian_3_square_sums(n)

print("sum_2_squares: {}".format(sum_2_squares))
print("sum_3_squares: {}".format(sum_3_squares))

x, y, z, w = zip(*sum_3_squares)

# plt.figure()

# plt.plot(x, y, "b.")
# plt.plot(x, z, "c.")
# plt.plot(y, z, "r.")

# plt.show()

pix = np.zeros((n, n, 3)).astype(np.uint8)

pix[y, x, 0] = 255
pix[z, y, 1] = 255
pix[z, x, 2] = 255

img = Image.fromarray(pix)
# img.show()
img.save("pythagorian_3_sums_n_{}.png".format(n), format="PNG")

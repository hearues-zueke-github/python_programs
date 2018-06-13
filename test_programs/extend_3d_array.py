#! /usr/bin/python3.5

import numpy as np

# First get the Matrix A
A = np.random.randint(0, 10, (3, 2, 4))

print("A:\n{}".format(A))

# Add 2 lines above and 1 line below for the matrix
# with the shape (2, 4), so it should have the (5, 4) shape
B = np.zeros(np.array(A.shape)+(0, 1+2, 0)).astype(A.dtype)



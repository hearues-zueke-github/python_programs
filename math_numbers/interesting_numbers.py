#! /usr/bin/python3.5

import numpy as np

if __name__ == "__main__":
    # get all possible shapes of arrays with numbers between 0-9 and also with the size 1-a and 1-b
    np.sum(10**np.multiply.outer(np.arange(1,4), np.arange(1, 7)).astype(object))

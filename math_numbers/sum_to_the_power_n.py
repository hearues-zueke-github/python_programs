#! /usr/bin/python2.7

import numpy as np

# a**2+b**2=c**2, find some numbers

if __name__ == "__main__":
    n_max = 100
    x = np.arange(1, n_max)

    nums_pow_2 = np.arange(1, n_max)**2
    pow_2_to_num = {i**2: i for i in x}

    pow_2_pairs = [(pow_2_to_num[a], pow_2_to_num[b], pow_2_to_num[a+b]) for a in nums_pow_2 for b in nums_pow_2 if a+b in nums_pow_2]

    print("pow_2_pairs:\n{}".format(pow_2_pairs))

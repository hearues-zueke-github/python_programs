#! /usr/bin/python3.5

import numpy as np

def own_pow(b, e):
    e_base2_inv = list(map(int, bin(e)[2:][::-1]))
    print("e_base2_inv: {}".format(e_base2_inv))
    
    p = 1
    a = b
    for idx, i in enumerate(e_base2_inv):
        if i == 1:
            p *= a
        print("idx: {}, i: {}, a: {}, p: {}".format(idx, i, a, p))
        a *= a

    return p

if __name__ == "__main__":
    b = 7
    e = 49

    true_number = b**e
    own_number = own_pow(b, e)

    print("true_number: {}".format(true_number))
    print("own_number:  {}".format(own_number))

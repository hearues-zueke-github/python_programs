#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import os
import sys


if __name__ == "__main__":
    for n in range(2, 9):
        for m_amount in range(2, n+1):
            str_cmd = "./rotation_1d_puzzle_solver.py {} {}".format(n, m_amount)
            os.system(str_cmd)
#! /usr/bin/python3

# -*- coding: utf-8 -*-

import multiprocessing as mp
from multiprocessing import Process

def void_func():
	while 1: pass

if __name__ == "__main__":
	cpu_amount = mp.cpu_count()
	ps = [Process(target=void_func) for _ in range(0, cpu_amount)]
	for p in ps: p.start()
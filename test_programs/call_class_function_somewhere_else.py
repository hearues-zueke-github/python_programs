#!/usr/bin/env python3

# -*- coding: utf-8 -*-
 
class A(Exception):
    def __init__(self):
        self.i = 0


    def inc(self):
        print("function 'inc' called")
        self.i += 1


    def dec(self):
        print("function 'dec' called")
        self.i -= 1


    def print_i(self):
        print("function 'print_i' called")
        print("self.i: {}".format(self.i))


if __name__=='__main__':
    print('Hello World!')

    a = A()
    
    i = a.inc
    d = a.dec
    p = a.print_i

    lst_f = [i, i, d, p, i, i, i, p, d, p, d, p]

    for f in lst_f:
        f()

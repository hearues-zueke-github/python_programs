#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

def foo(a, b):
    # print here the expression of a e.g., not the evaluated value!

    
    return a+b

if __name__ == "__main__":
    c = foo((2+3)-5, 1+2+3*(2+3))
    print(f"c: {c}")

#! /usr/bin/python3.5

# -*- coding: utf-8 -*-

l = [1, 2, 3]
# globals()["l"] = [1, 2, 3]
def g0():
    global l
    print("g0: outside l: {}".format(l))
    l = [2, 3, 4]
    def g1():
        global l
        print("g1: outside l: {}".format(l))
        l = [3, 4, 6]
        def g2():
            global l
            print("g2: outside l: {}".format(l))
            l = [4, 5, 8]
            print("g2: inside  l: {}".format(l))
        g2()
        print("g1: inside  l: {}".format(l))
    g1()
    print("g0: inside  l: {}".format(l))

if __name__ == "__main__":
    g0()

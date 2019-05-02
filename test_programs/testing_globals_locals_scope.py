#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

f3_str = """
def f3():
    print("c: {}".format(c))
    print("locals(): {}".format(locals()))
"""

class A():
    def __init__(self):
        pass


def f1():
    print("f1()")
    a = 2
    b = 3
    def f2():
        print("f2()")
        print("b: {}".format(b))
        print("locals(): {}".format(locals()))
        # print("globals(): {}".format(globals()))
    print("locals(): {}".format(locals()))
    # print("globals(): {}".format(globals()))
    return f2

if __name__ == "__main__":
    # f2 = f1()
    # f2()

    # loc = {'a':0}
    # exec(f3_str, globals(), loc)
    # f3 = loc['f3']
    # a = 3
    # f3()
    # # locals()['a'] = 4
    # # f3()
    # a = 4
    # loc['a'] = 2
    # # globals()['a'] = 5
    # f3()
    # # f1.locals['b'] = 4
    # # f1()

    a = A()
    print("a: {}".format(a))
    print("a.__dict__: {}".format(a.__dict__))

    a.c = 5
    exec(f3_str, a.__dict__)
    f3 = a.f3

    print("a.__dict__: {}".format(a.__dict__))
    f3()

    a.c = 7
    f3()

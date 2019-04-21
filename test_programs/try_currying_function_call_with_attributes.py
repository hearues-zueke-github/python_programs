#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

def get_g():
    def g(n=1):
        if n > 0:
            g.lst.append(g.i)
            g.i += 1
            g(n-1)
    g.i = 0
    g.lst = []
    return g


if __name__ == "__main__":
    g1 = get_g()
    g2 = get_g()
    g3 = get_g()

    g1(n=3)
    g2(n=5)
    g3(n=4)

    print("g1.i: {}, g1.lst: {}".format(g1.i, g1.lst))
    print("g2.i: {}, g2.lst: {}".format(g2.i, g2.lst))
    print("g3.i: {}, g3.lst: {}".format(g3.i, g3.lst))

    assert g1.i == 3
    assert g1.lst == [0, 1, 2]
    assert g2.i == 5
    assert g2.lst == [0, 1, 2, 3, 4]
    assert g3.i == 4
    assert g3.lst == [0, 1, 2, 3]

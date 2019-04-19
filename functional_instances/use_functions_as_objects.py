#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

if __name__ == "__main__":
    def get_a():
        def a():
            return get_a()

        a.i = 0
        a.lst = []

        def f():
            a.i += 1
            a.lst.append(a.i)
            return a.i
        a.f = f

        return a

    def get_b():
        def b():
            return get_b()

        b.i = 0
        b.arr = []
        b.lst = []

        def f(i=1):
            if i <= 0:
                b.arr = []
                return []

            arr = [j for j in range(b.i, b.i+i)]
            b.i += i
            b.lst.extend(arr)
            return arr
        b.f = f

        return b

    a = get_a()
    a2 = get_a()
    another_a = a()

    print("a.f(): {}".format(a.f()))
    print("a.f(): {}".format(a.f()))
    print("a.lst: {}".format(a.lst))
    
    print("a2.f(): {}".format(a2.f()))
    print("a2.f(): {}".format(a2.f()))
    print("a2.f(): {}".format(a2.f()))
    print("a2.lst: {}".format(a2.lst))

    print("another_a.f(): {}".format(another_a.f()))
    print("another_a.f(): {}".format(another_a.f()))
    print("another_a.f(): {}".format(another_a.f()))
    print("another_a.f(): {}".format(another_a.f()))
    print("another_a.lst: {}".format(another_a.lst))

    b = get_b()
    print("b.f(): {}".format(b.f()))
    print("b.f(1): {}".format(b.f(1)))
    print("b.f(i=0): {}".format(b.f(i=0)))
    print("b.f(i=5): {}".format(b.f(i=5)))

    print("b.arr: {}".format(b.arr))
    print("b.lst: {}".format(b.lst))

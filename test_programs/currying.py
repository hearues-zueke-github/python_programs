#! /usr/bin/python3.5

def f(a):
    def g(b):
        return a+b
    return g

f1 = f(3)
f2 = f(6)

print("f1(4): {}".format(f1(4)))
print("f2(4): {}".format(f2(4)))

print("f(3)(4): {}".format(f(3)(4)))
print("f(6)(4): {}".format(f(6)(4)))

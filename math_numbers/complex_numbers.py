#! /usr/bin/python2.7

import sys

import numpy as np

get_random_num = lambda: np.random.randint(1, 11)

# print("get_random_num(): {}".format(get_random_num()))
# sys.exit(0)

def gcd(a, b):
    if a == b:
        return a
    if a < b:
        a, b = b, a

    while b > 0:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a*b/gcd(a, b)

class Fraction(Exception):
    def __init__(self, a, b):
        if b == 0:
            raise ZeroDivisionError("divisor is zero")
        factor = gcd(abs(a), b)
        self.a = a//factor
        self.b = b//factor

    def __str__(self):
        return "frac({}, {})".format(self.a, self.b)

    def __neg__(self):
        return Fraction(-self.a, self.b)

    def __pow__(self, other):
        if not isinstance(other, int):
            raise ValueError("other must be type of int, but it is type of {}".format(type(other)))
        return Fraction(self.a**2, self.b**2)

    def __add__(self, other):
        common_div = lcm(self.b, other.b)
        x  = self.a*common_div//self.b
        y  = other.a*common_div//other.b
        return Fraction(x+y, common_div)

    def __sub__(self, other):
        common_div = lcm(self.b, other.b)
        x  = self.a*common_div//self.b
        y  = other.a*common_div//other.b
        return Fraction(x-y, common_div)

    def __mul__(self, other):
        if not isinstance(other, Fraction):
            raise ValueError("other must be type of Fraction, but it is type of {}".format(type(other)))
        return Fraction(self.a*other.a, self.b*other.b)

    def __div__(self, other):
        if not isinstance(other, Fraction):
            raise ValueError("other must be type of Fraction, but it is type of {}".format(type(other)))
        if other.a == 0:
           raise ZeroDivisionError("divisor is zero")
        return Fraction(self.a*other.b, self.b*other.a)

class Complex(Exception):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def conj(self):
        return Complex(self.a, -self.b)

    def __str__(self):
        return "({}, {})".format(self.a, self.b)

    def __add__(self, other):
        if not isinstance(other, Complex):
            raise ValueError("other must be type of Complex, but it is type of {}".format(type(other)))
        return Complex(self.a+other.a, self.b+other.b)

    def __sub__(self, other):
        if not isinstance(other, Complex):
            raise ValueError("other must be type of Complex, but it is type of {}".format(type(other)))
        return Complex(self.a-other.a, self.b-other.b)

    def __mul__(self, other):
        if not isinstance(other, Complex):
            raise ValueError("other must be type of Complex, but it is type of {}".format(type(other)))
        return Complex(self.a*other.a-self.b*other.b, self.a*other.b+self.b*other.a)

    def __div__(self, other):
        if not isinstance(other, Complex):
            raise ValueError("other must be type of Complex, but it is type of {}".format(type(other)))
        if (other.a == 0 or other.a == 0.) and \
           (other.b == 0 or other.b == 0.):
           raise ZeroDivisionError("divisor is zero")
        new = self*other.conj()
        divisor = other.a**2+other.b**2
        return Complex(new.a/divisor, new.b/divisor)

a = 15
b = 4

print("a: {}".format(a))
print("b: {}".format(b))
print("gcd(a, b): {}".format(gcd(a, b)))

f1 = Fraction(1, 2)
f2 = Fraction(4, 6)

print("f1: {}".format(f1))
print("f2: {}".format(f2))

print("f1+f2: {}".format(f1+f2))
print("f1-f2: {}".format(f1-f2))
print("f1*f2: {}".format(f1*f2))
print("f1/f2: {}".format(f1/f2))

# sys.exit(0)

a1 = -3
b1 = 4
a2 = 5
b2 = 2

c1 = Complex(a1, b1)
c2 = Complex(a2, b2)

frac_a1 = Fraction(get_random_num(), get_random_num())
frac_b1 = Fraction(get_random_num(), get_random_num())
frac_a2 = Fraction(get_random_num(), get_random_num())
frac_b2 = Fraction(get_random_num(), get_random_num())

cf1 = Complex(frac_a1, frac_b1)
cf2 = Complex(frac_a2, frac_b2)

print("cf1: {}".format(cf1))
print("cf2: {}".format(cf2))

print("cf1+cf2: {}".format(cf1+cf2))
print("cf1-cf2: {}".format(cf1-cf2))
print("cf1*cf2: {}".format(cf1*cf2))
print("cf1/cf2: {}".format(cf1/cf2))

print("c1: {}".format(c1))
print("c2: {}".format(c2))

print("c1+c2: {}".format(c1+c2))

print("c1.conj(): {}".format(c1.conj()))
print("c1*c2: {}".format(c1*c2))

print("c1*c1.conj(): {}".format(c1*c1.conj()))

print("c1/c2: {}".format(c1/c2))

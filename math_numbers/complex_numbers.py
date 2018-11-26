#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import sys

import numpy as np

get_random_num = lambda: np.random.randint(1, 11)

def gcd(a, b):
    if a == b:
        return a
    if a < b:
        a, b = b, a

    while b > 0:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a*b//gcd(a, b)

class Fraction(Exception):
    def __init__(self, a, b):
        self.a_type = type(a)
        self.b_type = type(b)
        
        if b == 0:
            raise ZeroDivisionError("divisor is zero")
        factor = gcd(abs(a), b)
        self.a = a//factor
        self.b = b//factor

    def rec(self):
        if isinstance(self.a, int) and self.a == 0:
            raise ZeroDivisionError("divisor is zero")
        return Fraction(self.b, self.a)

    def __str__(self):
        return "frac({}, {})".format(self.a, self.b)

    def __neg__(self):
        return Fraction(-self.a, self.b)

    def __pow__(self, other):
        if isinstance(other, int):
            return Fraction(self.a**2, self.b**2)
        
        raise ValueError("other must be type of int, but it is type of {}".format(type(other)))

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

    def __floordiv__(self, other):
        if not isinstance(other, Fraction):
            raise ValueError("other must be type of Fraction, but it is type of {}".format(type(other)))
        
        if other.a == 0:
           raise ZeroDivisionError("divisor is zero")
        return Fraction(self.a*other.b, self.b*other.a)

class Complex(Exception):
    def __init__(self, a, b):
        if not isinstance(a, Fraction):
            raise ValueError("a is not type Fraction! a is type of {}!".format(type(a)))
        if not isinstance(b, Fraction):
            raise ValueError("b is not type Fraction! b is type of {}!".format(type(b)))
        
        # print("instantation of Complex!")
        # print("a: {}".format(a))
        # print("b: {}".format(b))

        self.a = a
        self.b = b

    def conj(self):
        return Complex(self.a, -self.b)

    def __str__(self):
        return "({}, {})".format(self.a, self.b)

    def __add__(self, other):
        if isinstance(other, Complex):
            return Complex(self.a+other.a, self.b+other.b)
        raise ValueError("other must be type of Complex, but it is type of {}".format(type(other)))

    def __sub__(self, other):
        if isinstance(other, Complex):
            return Complex(self.a-other.a, self.b-other.b)
        raise ValueError("other must be type of Complex, but it is type of {}".format(type(other)))

    def __mul__(self, other):
        if isinstance(other, Complex):
            return Complex(self.a*other.a-self.b*other.b, self.a*other.b+self.b*other.a)
        raise ValueError("other must be type of Complex, but it is type of {}".format(type(other)))

    def __floordiv__(self, other):
        if not isinstance(other, Complex):
            raise ValueError("other must be type of Complex, but it is type of {}".format(type(other)))

        if (other.a == 0 or other.a == 0.) and \
           (other.b == 0 or other.b == 0.):
           raise ZeroDivisionError("divisor is zero")
        new = self*other.conj()
        divisor = other.a**2+other.b**2

        return Complex(new.a//divisor, new.b//divisor)


# a = 15
# b = 4

# print("a: {}".format(a))
# print("b: {}".format(b))
# print("gcd(a, b): {}".format(gcd(a, b)))

# f1 = Fraction(1, 2)
# f2 = Fraction(4, 6)

# print("f1: {}".format(f1))
# print("f2: {}".format(f2))

# print("f1+f2: {}".format(f1+f2))
# print("f1-f2: {}".format(f1-f2))
# print("f1*f2: {}".format(f1*f2))
# print("f1//f2: {}".format(f1//f2))
# print("f1.rec(): {}".format(f1.rec()))
# print("f2.rec(): {}".format(f2.rec()))

# sys.exit(0)

# a1 = -3
# b1 = 4
# a2 = 5
# b2 = 2

# c1 = Complex(a1, b1)
# c2 = Complex(a2, b2)

frac_a1 = Fraction(get_random_num(), get_random_num())
frac_b1 = Fraction(get_random_num(), get_random_num())
frac_a2 = Fraction(get_random_num(), get_random_num())
frac_b2 = Fraction(get_random_num(), get_random_num())

print("frac_a1: {}".format(frac_a1))
print("frac_b1: {}".format(frac_b1))
print("frac_a2: {}".format(frac_a2))
print("frac_b2: {}".format(frac_b2))

cf1 = Complex(frac_a1, frac_b1)
cf2 = Complex(frac_a2, frac_b2)

print("cf1: {}".format(cf1))
print("cf2: {}".format(cf2))

print("cf1+cf2: {}".format(cf1+cf2))
print("cf1-cf2: {}".format(cf1-cf2))
print("cf1*cf2: {}".format(cf1*cf2))
print("cf1/cf2: {}".format(cf1//cf2))

# print("c1: {}".format(c1))
# print("c2: {}".format(c2))

# print("c1+c2: {}".format(c1+c2))

# print("c1.conj(): {}".format(c1.conj()))
# print("c1*c2: {}".format(c1*c2))

# print("c1*c1.conj(): {}".format(c1*c1.conj()))

# print("c1/c2: {}".format(c1//c2))

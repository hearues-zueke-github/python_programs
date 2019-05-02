import numpy as np

import decimal
decimal.getcontext().prec = 100

from decimal import Decimal as Dec

class ComplexDec(Exception):
    def __init__(self, a, b):
        self.a = Dec(a)
        self.b = Dec(b)

    def __str__(self):
        return "({a}{sign}{b}*j)".format(a=self.a, sign="" if self.b < 0 else "+", b=self.b)

    def __add__(self, other):
        a1 = self.a
        b1 = self.b
        if isinstance(other, ComplexDec):
            a2 = other.a
            b2 = other.b
        elif isinstance(other, complex):
            a2 = Dec(other.real)
            b2 = Dec(other.imag)
        else:
            a2 = Dec(other)
            b2 = Dec(0)
        return ComplexDec(a1+a2, b1+b2)

    def __sub__(self, other):
        a1 = self.a
        b1 = self.b
        if isinstance(other, ComplexDec):
            a2 = other.a
            b2 = other.b
        elif isinstance(other, complex):
            a2 = Dec(other.real)
            b2 = Dec(other.imag)
        else:
            a2 = Dec(other)
            b2 = Dec(0)
        return ComplexDec(a1-a2, b1-b2)

    def __mul__(self, other):
        a1 = self.a
        b1 = self.b
        if isinstance(other, ComplexDec):
            a2 = other.a
            b2 = other.b
        elif isinstance(other, complex):
            a2 = Dec(other.real)
            b2 = Dec(other.imag)
        else:
            a2 = Dec(other)
            b2 = Dec(0)
        return ComplexDec(a1*a2-b1*b2, a1*b2+a2*b1)

    def __floordiv__(self, other):
        a1 = self.a
        b1 = self.b
        if isinstance(other, ComplexDec):
            a2 = other.a
            b2 = other.b
        elif isinstance(other, complex):
            a2 = Dec(other.real)
            b2 = Dec(other.imag)
        else:
            a2 = Dec(other)
            b2 = Dec(0)
        divisor = a2**2+b2**2
        return ComplexDec((a1*a2+b1*b2)/divisor, (-a1*b2+a2*b1)/divisor)

    def __truediv__(self, other):
        a1 = self.a
        b1 = self.b
        if isinstance(other, ComplexDec):
            a2 = other.a
            b2 = other.b
        elif isinstance(other, complex):
            a2 = Dec(other.real)
            b2 = Dec(other.imag)
        else:
            a2 = Dec(other)
            b2 = Dec(0)
        divisor = a2**2+b2**2
        return ComplexDec((a1*a2+b1*b2)/divisor, (-a1*b2+a2*b1)/divisor)

    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __rfloordiv__(self, other):
        return self.__floordiv__(other)
    
    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def abs(self):
        return (self.a**2+self.b**2)**Dec("0.5")

    def arg(self):
        return np.arctan(float(self.b/self.a))


def test_cplx_nums():
    print("Now calling method 'test_cplx_nums'!")

    print("ComplexDec class:")
    c1 = ComplexDec(2, 3)
    c2 = ComplexDec(3, 4.6)
    print("  c1: {}".format(c1))
    print("  c2: {}".format(c2))
    print("  c1+c2: {}".format(c1+c2))
    print("  c1-c2: {}".format(c1-c2))
    print("  c1*c2: {}".format(c1*c2))
    print("  c1/c2: {}".format(c1/c2))
    print("  c1//c2: {}".format(c1/c2))

    print("complex class:")
    cc1 = complex(2, 3)
    cc2 = complex(3, 4.6)
    print("  cc1: {}".format(cc1))
    print("  cc2: {}".format(cc2))
    print("  cc1+cc2: {}".format(cc1+cc2))
    print("  cc1-cc2: {}".format(cc1-cc2))
    print("  cc1*cc2: {}".format(cc1*cc2))
    print("  cc1/cc2: {}".format(cc1/cc2))
    print("  cc1//cc2: {}".format(cc1/cc2))

if __name__ == "__main__":
    test_cplx_nums()

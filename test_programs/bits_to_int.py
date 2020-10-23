#! /usr/bin/python3

import timeit

bit_list = [1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1,0,0,0,0,0,0,1]

def mult_and_add(bit_list):
    output = 0
    for bit in bit_list:
        output = output * 2 + bit
    return output

def shifting(bitlist):
     out = 0
     for bit in bitlist:
         out = (out << 1) | bit
         # out = (out << 1) | bit
     return out

def summing(bitlist):
    return sum([b<<i for i, b in enumerate(bitlist)])

def summing_reverse(bitlist):
    return sum([b<<i for i, b in enumerate(bitlist[::-1])])


n = 1000000

t1 = timeit.timeit('convert(bit_list)', 'from __main__ import mult_and_add as convert, bit_list', number=n)
print("mult and add method time is : {} ".format(t1))
t2 = timeit.timeit('convert(bit_list)', 'from __main__ import shifting as convert, bit_list', number=n)
print("shifting method time is : {} ".format(t2))
t3 = timeit.timeit('convert(bit_list)', 'from __main__ import summing as convert, bit_list', number=n)
print("summing method time is : {} ".format(t3))
t4 = timeit.timeit('convert(bit_list)', 'from __main__ import summing_reverse as convert, bit_list', number=n)
print("summing_reverse method time is : {} ".format(t4))

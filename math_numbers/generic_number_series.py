#! /usr/bin/python2.7

import numpy as np

import matplotlib.pyplot as plt

# k(n) = round up (n/2)
# x(i, n, a_n) = (i + n + a_n) mod y, where y is a natural number

a = [0, 0, 0, 0, 1]

y = 7

for n in xrange(len(a), 10000):
    #k = int(np.ceil(np.sqrt(n)))#n/9.))
    #k = 3
    k = 5
    a_sum = 0
    #y=n
    last = len(a)
    for i in xrange(1, k+1):
        #a_sum += (i + n + a[last-i]) % y
        a_sum += (a[last-i]+n) % y
    a.append(a_sum)

print("y = {}".format(y))

for i, ai in enumerate(a):
    #if ai == 1:
    print("i = {}  a[{}] = {}".format(i, i, ai))

plt.plot(np.arange(len(a)), a, ".b")
plt.show()

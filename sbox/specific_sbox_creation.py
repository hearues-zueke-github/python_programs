#! /usr/bin/python3.6

import numpy as np

def get_next_array(n, a, x):
    s = np.cumsum(a) % n
    s = np.hstack((s[1:], [s[0]]))
    x = list(x)
    # x = x.tolist()
    y = []
    acc = 0
    j = n
    for i in s:
        acc = (acc+i)%j
        j -= 1
        y.append(x.pop(acc))

    return np.array(y)

if __name__ == "__main__":
    n = 4
    a0 = np.arange(0, n)

    print("a0: {}".format(a0))

    # a1 = get_next_array(n, a0, a0)
    # print("a1: {}".format(a1))

    # a20 = get_next_array(n, a1, a0)
    # a21 = get_next_array(n, a0, a1)

    # print("a20: {}".format(a20))
    # print("a21: {}".format(a21))

    found_lsts = 1
    lsts = [a0.tolist()]
    combos = [[]]
    # combos = [[(-1, 0)]]
    a1 = a0.copy()
    for idx in range(1, 100001):
        i0 = np.random.randint(0, found_lsts)
        i1 = np.random.randint(0, found_lsts)
        
        a0 = lsts[i0]
        a1 = lsts[i1]

        a2 = get_next_array(n, a1, a0).tolist()

        if a2 in lsts:
            i = lsts.index(a2)
            l_comb = combos[i]
            comb = (i1, i0)
            if not comb in l_comb:
                l_comb.append(comb)
                print("idx: {}".format(i))
            continue
        print("idx: {}, a1: {}".format(idx, a1))

        lsts.append(a2)
        combos.append([(i1, i0)])
        found_lsts += 1

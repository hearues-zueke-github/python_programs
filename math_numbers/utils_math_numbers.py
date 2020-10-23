def convert_n_to_other_base(n, b):
    if n==0:
        return [0]
    elif b==1:
        return [1]*n
    l = []
    while n>0:
        l.append(n%b)
        n //= b
    return list(reversed(l))


def convert_base_n_to_num(l, b):
    s = 0
    k = 1
    for v in l:
        s += v*k
        k *= b
    return s

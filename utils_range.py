def range_gen_old(g, n):
    try:
        i = 0
        if i >= n:
            return
        v = next(g)
        while v != None:
            yield v
            v = next(g)
            i += 1
            if i >= n:
                break
    except StopIteration:
        return

def range_gen(g, n):
    try:
        i = 0
        while ((i := i + 1) <= n) and ((v := next(g)) != None):
            yield v
    except StopIteration:
        return

def gen_new_gen_nr1():
    return iter(range(0, 100))

l1 = list(range_gen_old(gen_new_gen_nr1(), 0))
l2 = list(range_gen(gen_new_gen_nr1(), 0))
assert l1 == l2

l1 = list(range_gen_old(gen_new_gen_nr1(), 1))
l2 = list(range_gen(gen_new_gen_nr1(), 1))
assert l1 == l2

l1 = list(range_gen_old(gen_new_gen_nr1(), 40))
l2 = list(range_gen(gen_new_gen_nr1(), 40))
assert l1 == l2

l1 = list(range_gen_old(gen_new_gen_nr1(), 200))
l2 = list(range_gen(gen_new_gen_nr1(), 200))
assert l1 == l2

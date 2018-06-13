#! /usr/bin/python3.5

# First append the list a with itself! (this is recursive)
a = []
a.append(a)

get_str_expr = lambda n: "a"+"[0]"*n

increment = 1000
# first find n_max
n_prev = 0
n = 1000

while increment > 0:
    n = n_prev
    while True:
        try:
            eval(get_str_expr(n))
        except:
            break
        n_prev = n
        n += increment

    print("list a: n_prev: {}".format(n_prev))
    print("list a: n: {}".format(n))

    increment //= 10

# Return the function by itself!
f = lambda: f

get_func_str_expr = lambda n: "f"+"()"*n

increment = 1000
# first find n_max
n_prev = 0
n = 1000

while increment > 0:
    n = n_prev
    while True:
        try:
            eval(get_func_str_expr(n))
        except:
            break
        n_prev = n
        n += increment

    print("function f: n_prev: {}".format(n_prev))
    print("function f: n: {}".format(n))

    increment //= 10

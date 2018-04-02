#! /usr/bin/python2.7

"""
    patterns for:
    n = 1
    (x)
    n = 2
    (x,x); (x|x)
    n = 3
    (x,x,x); (x,x|x); (x,(x|x)); (x|x|x)
    n = 4
    (x,x,x,x);
    (x,x,x|x); (x,x,(x|x)); (x,(x,x|x)); (x,x|x,x);
    (x,x|x|x); (x,(x|x|x)); (x|(x,x|x));
    (x|x|x|x)
    n = 5
    (x,x,x,x,x);
    (x,x,x,x|x); (x,x,x,(x|x)); (x,x,(x,x|x)); (x,(x,x,x|x)); (x,x,x|x,x); (x,(x,x|x,x));
    (x,x,x|x|x); (x,x,(x|x|x)); (x,(x,x|x|x)); (x,x|x,x|x); (x,x|x,(x|x)); (x,(x|x,(x|x))); (x,(x|x),(x|x));
    (x,x|x|x|x); (x,(x|x|x|x)); ((x|x),(x|x|x))
    (x|x|x|x|x)
"""

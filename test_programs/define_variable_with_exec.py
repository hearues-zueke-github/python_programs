#! /usr/bin/python3.6

import ast
import pdb

def test_exec_in_a_function():
    # tree = ast.parse("c = 6") 
    # exec(compile(tree, filename="<ast>", mode="exec"))
    # exec("b = 5", locals())
    exec("globals()['c'] = 5")
    exec("globals()['b'] = 7")

    # print("globals()['c']: {}".format(globals()["c"]))
    # print("globals()['b']: {}".format(globals()["b"]))

    # pdb.set_trace()
    print("c: {}".format(c))
    print("b: {}".format(b))
    
if __name__ == "__main__":
    exec("a = 5")
    print("a: {}".format(a))
    
    tree = ast.parse("d = 2") 
    exec(compile(tree, filename="<ast>", mode="exec"))
    print("d: {}".format(d))

    test_exec_in_a_function()

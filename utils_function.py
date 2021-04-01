from types import FunctionType

def copy_function(f, d_glob={}):
    return FunctionType(f.__code__, d_glob, f.__name__, f.__defaults__, f.__closure__)

import importlib.util as imp_util

def load_module_dynamically(var_glob, name, path):
    spec = imp_util.spec_from_file_location(name, path)
    module = imp_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    var_glob[name] = module

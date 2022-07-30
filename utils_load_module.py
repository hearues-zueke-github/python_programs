import ast
import sys # is needed for globally defining imported modules

import importlib.machinery
import importlib.util

def load_module_dynamically(var_glob, name, path):
	if name in sys.modules:
		var_glob[name] = sys.modules[name]
		return

	loader = importlib.machinery.SourceFileLoader(name, path)
	spec = importlib.util.spec_from_loader(name, loader)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)

	sys.modules[name] = module
	var_glob[name] = module

import os

import importlib.util as imp_util

from typing import List, Tuple, Dict, Any

def load_imports_template(
    var_globals: Dict[str, Any],
    file_path: str,
    l_attribute: List[str],
) -> None:
    assert os.path.exists(file_path)
    spec = imp_util.spec_from_file_location("module", file_path)
    module = imp_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module_dict = module.__dict__
    for attribute in l_attribute:
        var_globals[attribute] = module_dict[attribute]

def load_imports(
    var_globals: Dict[str, Any],
    l_file_path_l_attribute: List[Tuple[str, List[str]]],
) -> List[Tuple[str, List[str]]]:
    for file_path, l_attribute in l_file_path_l_attribute:
        load_imports_template(var_globals, file_path, l_attribute)

    return l_file_path_l_attribute

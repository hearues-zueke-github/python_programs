import os

from typing import List, Dict, Set, Mapping, Any, Union

def mkdirs(path : str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

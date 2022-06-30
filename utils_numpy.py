import numpy as np

from typing import List, Dict, Set, Mapping, Any, Union, Tuple

def unique(l : Union[np.ndarray, List[Any]]) -> np.ndarray:
	return np.unique(l)


def unique_counts(l : Union[np.ndarray, List[Any]]) -> Tuple[np.ndarray, np.ndarray]:
	return np.unique(l, return_counts=True)

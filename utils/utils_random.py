import hashlib

import numpy as np

from datetime import datetime

def get_current_datetime_str() -> str:
	return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')

def get_random_seed_256_bits(max_round=3) -> np.ndarray:
	b: bytearray = hashlib.sha256().digest()

	for _ in range(0, max_round):
		m: hashlib.sha256 = hashlib.sha256()
		m.update(b + get_current_datetime_str().encode('utf-8'))
		b = m.digest()
	
	return np.array(list(b), dtype=np.uint8)

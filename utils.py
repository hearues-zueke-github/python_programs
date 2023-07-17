import os
import platform
import subprocess
import re

from hashlib import sha256
from typing import List, Dict, Set, Mapping, Any, Union

BLOCK_SIZE = sha256().block_size * 2**14

def mkdirs(path : str) -> None:
	if not os.path.exists(path):
		os.makedirs(path)


def path_exists(path: str) -> bool:
	return os.path.exists(path)


# from stackoverflow: https://stackoverflow.com/questions/4842448/getting-processor-information-in-python
def get_processor_name():
	if platform.system() == "Windows":
		return platform.processor()
	elif platform.system() == "Darwin":
		os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
		command ="sysctl -n machdep.cpu.brand_string"
		return subprocess.check_output(command).strip()
	elif platform.system() == "Linux":
		command = "cat /proc/cpuinfo"
		all_info = subprocess.check_output(command, shell=True).decode().strip()
		for line in all_info.split("\n"):
			if "model name" in line:
				return re.sub( ".*model name.*: ", "", line,1)
	return ""


def hash_file_sha256(file_path: str) -> str:
	sha256_obj = sha256()

	with open(file_path, 'rb') as f:
		block = f.read(BLOCK_SIZE)
		
		while block:
			sha256_obj.update(block)
			block = f.read(BLOCK_SIZE)

	hash_sum = sha256_obj.digest()

	return hash_sum

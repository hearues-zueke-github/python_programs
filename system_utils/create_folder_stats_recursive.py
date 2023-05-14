#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.10 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import datetime
import dill
import gzip
import hashlib
import os
import pdb
import re
import sys
import traceback
import tarfile

import numpy as np
import pandas as pd
import multiprocessing as mp

from collections import defaultdict
from copy import deepcopy, copy
from dotmap import DotMap
from functools import reduce
from hashlib import sha256
from io import BytesIO
from memory_tempfile import MemoryTempfile
from shutil import copyfile
from pprint import pprint
from typing import List, Set, Tuple, Dict, Union, Any
from PIL import Image
from recordclass import RecordClass, asdict

CHUNK_SIZE = 1048576 # 2**20 = 1MiB

class RootDirsFiles(RecordClass):
	root: str
	l_dir_name: List[str]
	l_file_name: List[str]

CURRENT_WORKING_DIR = os.getcwd()
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = MemoryTempfile().gettempdir()
PYTHON_PROGRAMS_DIR = os.path.join(HOME_DIR, 'git/python_programs')

# set the relative/absolute path where the utils_load_module.py file is placed!
sys.path.append(PYTHON_PROGRAMS_DIR)
from utils_load_module import load_module_dynamically

var_glob = globals()
load_module_dynamically(**dict(var_glob=var_glob, name='utils', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_multiprocessing_manager', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils_multiprocessing_manager.py")))

mkdirs = utils.mkdirs
MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

OBJS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'objs')
mkdirs(OBJS_DIR_PATH)

PLOTS_DIR_PATH = os.path.join(PATH_ROOT_DIR, 'plots')
mkdirs(PLOTS_DIR_PATH)

def get_df_dir_file_stats(src_folder_path):
	l_dir_path_link = []
	l_file_path_link = []

	# first walk through all the dirs and files
	l_rootdirfile = [RootDirsFiles(*next(os.walk(src_folder_path)))]
	l_rootdirfile_temp = [rootdirfile for rootdirfile in l_rootdirfile]

	iter_nr = 0
	while len(l_rootdirfile_temp) > 0:
		iter_nr += 1
		print("iter_nr: {}".format(iter_nr))

		len_l_rootdirfile_temp = len(l_rootdirfile_temp)
		l_rootdirfile_temp_2 = []
		for i_rootdirfile, rootdirfile in enumerate(l_rootdirfile_temp, 1):
			print("- rootdirfile {:5}/{:5}".format(i_rootdirfile, len_l_rootdirfile_temp))
			root = rootdirfile.root
			for dir_name in rootdirfile.l_dir_name:
				dir_path = os.path.join(root, dir_name)
				if os.path.islink(dir_path):
					l_dir_path_link.append(dir_path)
					print('-- path is link! "{}"'.format(dir_path))
					continue
				l_rootdirfile_temp_2.append(RootDirsFiles(*next(os.walk(dir_path))))

		l_rootdirfile_temp = l_rootdirfile_temp_2
		l_rootdirfile.extend(l_rootdirfile_temp)


	# then obtain all stats of each dirs and files
	root_first = l_rootdirfile[0].root
	len_root_first = len(root_first) + 1
	
	df_dir_file = pd.DataFrame(data=[], columns=['root_first', 'rel_root', 'df_dir', 'df_file', 'amount_files'], dtype=object)
	row_nr_dir_file = 0

	len_l_rootdirfile = len(l_rootdirfile)
	for i_rootdirfile, rootdirfile in enumerate(l_rootdirfile, 1):
		print("rootdirfile nr. {:5}/{:5}".format(i_rootdirfile, len_l_rootdirfile))
		root = rootdirfile.root
		rel_root = root[len_root_first:]

		df_dir_part = pd.DataFrame(data=[], columns=['name', 'os.stat'], dtype=object)
		df_file_part = pd.DataFrame(data=[], columns=['name', 'os.stat', 'sha256sum', 'sha3_256sum'], dtype=object)
		row_nr_dir_part = 0
		row_nr_file_part = 0

		for dir_name in rootdirfile.l_dir_name:
			dir_path = os.path.join(root, dir_name)
			stat = os.stat(dir_path)
			
			df_dir_part.loc[row_nr_dir_part] = [dir_name, stat]
			row_nr_dir_part += 1

		for file_name in rootdirfile.l_file_name:
			file_path = os.path.join(root, file_name)

			if os.path.islink(file_path):
				l_file_path_link.append(file_path)
				continue

			stat = os.stat(file_path)

			df_file_part.loc[row_nr_file_part] = [file_name, stat, '', '']
			row_nr_file_part += 1

		df_dir_file.loc[row_nr_dir_file] = [root_first, rel_root, df_dir_part, df_file_part, df_file_part.shape[0]]
		row_nr_dir_file += 1

	return df_dir_file, l_dir_path_link, l_file_path_link


def calculate_hash_sums(df_dir_file):
	amount_files_total = np.sum(df_dir_file['amount_files'].values)
	acc_file_nr = 0
	for dir_file_nr, (row_nr_dir, row_dir) in enumerate(df_dir_file.iterrows(), 1):
		root_first = row_dir['root_first']
		rel_root = row_dir['rel_root']

		df_file = row_dir['df_file']
		amount_files_part = df_file.shape[0]
		for file_nr, (row_nr, row) in enumerate(df_file.iterrows(), 1):
			name = row['name']
			rel_file_path = os.path.join(rel_root, name)
			file_path = os.path.join(root_first, rel_file_path)

			acc_file_nr += 1
			print("file_nr: {:5}/{:5}, total: {:6}/{:6} copy from {}".format(file_nr, amount_files_part, acc_file_nr, amount_files_total, file_path))

			try:
				h_sha256 = hashlib.sha256()
				h_sha3_256 = hashlib.sha3_256()

				with open(file_path, 'rb') as f:
					while True:
						data = f.read(CHUNK_SIZE)
						if not data:
							break
						h_sha256.update(data)
						h_sha3_256.update(data)

				df_file.loc[row_nr]['sha256sum'] =  h_sha256.hexdigest()
				df_file.loc[row_nr]['sha3_256sum'] =  h_sha3_256.hexdigest()
			except:
				df_file.loc[row_nr]['sha256sum'] =  ''
				df_file.loc[row_nr]['sha3_256sum'] =  ''


def calculate_hash_sums_if_diff_only(df_dir_file_now, df_dir_file_prev):
	# TODO: better would be to check, which files have the same st_ino number, so that
	#       the checksum should not be calculated again for the same file. Which is in a
	#       different folder after the file moving. Also the last modification date should be the same
	#       (as least).
	#       check if st_ino, st_ino and st_mtime are the same
	# check first, which root folders are the same and which are not

	df_merge = pd.merge(df_dir_file_now, df_dir_file_prev, how='left', on=['root_first', 'rel_root'], suffixes=('_now', '_prev'))
	df_merge = df_merge.where(pd.notnull(df_merge), None)

	amount_files_total = np.sum(df_dir_file['amount_files'].values)
	acc_file_nr = 0
	for dir_file_nr, (row_nr_dir, row_dir) in enumerate(df_merge.iterrows(), 1):
		root_first = row_dir['root_first']
		rel_root = row_dir['rel_root']

		df_file_now = row_dir['df_file_now']
		df_file_prev = row_dir['df_file_prev']

		if df_file_prev is None:
			df_file_prev = pd.DataFrame(data=[], columns=['name', 'os.stat', 'sha256sum', 'sha3_256sum'], dtype=object)

		df_merge_2 = pd.merge(df_file_now.reset_index(), df_file_prev, how='left', on=['name'], suffixes=('_now', '_prev'))
		df_merge_2 = df_merge_2.where(pd.notnull(df_merge_2), None)

		amount_files_part = df_file_now.shape[0]
		for file_nr, (row_nr, row) in enumerate(df_merge_2.iterrows(), 1):
			index = row['index']
			name = row['name']

			rel_file_path = os.path.join(rel_root, name)
			file_path = os.path.join(root_first, rel_file_path)

			acc_file_nr += 1
			print("file_nr: {:5}/{:5}, total: {:6}/{:6} copy from {}".format(file_nr, amount_files_part, acc_file_nr, amount_files_total, file_path))

			stat_now = row['os.stat_now']
			stat_prev = row['os.stat_prev']

			row_now = df_file_now.loc[index]

			if stat_prev is not None:
				if (
					(stat_now.st_ino == stat_prev.st_ino) and
					(stat_now.st_size == stat_prev.st_size) and
					(stat_now.st_mtime == stat_prev.st_mtime)
				):
					row_now['sha256sum'] = row['sha256sum_prev']
					row_now['sha3_256sum'] = row['sha3_256sum_prev']
					continue

			try:
				h_sha256 = hashlib.sha256()
				h_sha3_256 = hashlib.sha3_256()

				with open(file_path, 'rb') as f:
					while True:
						data = f.read(CHUNK_SIZE)
						if not data:
							break
						h_sha256.update(data)
						h_sha3_256.update(data)

				row_now['sha256sum'] =  h_sha256.hexdigest()
				row_now['sha3_256sum'] =  h_sha3_256.hexdigest()
			except:
				row_now['sha256sum'] =  ''
				row_now['sha3_256sum'] =  ''


if __name__ == '__main__':
	argv = sys.argv
	src_folder_path = argv[1]
	# file_path_folder_name_df_stats = argv[2]

	src_folder_path = src_folder_path.rstrip('/')

	# TODO: add possibility for checking in advance, if the file can be created (if the file is not already created)
	file_path_df_dir_file = src_folder_path+'.df_dir_file_stats.pkl.gz'
	# file_path_df_dir_file = file_path_folder_name_df_stats+'.df_dir_file_stats.pkl.gz'
	# file_path_df_dir_file = os.path.join(TEMP_DIR, file_path_folder_name_df_stats+'.df_dir_file_stats.pkl.gz')

	assert os.path.exists(src_folder_path)
	assert os.path.isdir(src_folder_path)
	assert not os.path.islink(src_folder_path)

	df_dir_file, l_dir_path_link, l_file_path_link = get_df_dir_file_stats(src_folder_path=src_folder_path)

	# and last step save files in dirs in the tar.gz, with the stats seperated + update the sha256sum and sha3_256
	if not os.path.exists(file_path_df_dir_file):
		calculate_hash_sums(df_dir_file=df_dir_file)
		
		with gzip.open(file_path_df_dir_file, 'wb') as f:
			dill.dump(df_dir_file, f)
	else:
		with gzip.open(file_path_df_dir_file, 'rb') as f:
			df_dir_file_prev = dill.load(f)
		
		calculate_hash_sums_if_diff_only(df_dir_file_now=df_dir_file, df_dir_file_prev=df_dir_file_prev)

		with gzip.open(file_path_df_dir_file, 'wb') as f:
			dill.dump(df_dir_file, f)

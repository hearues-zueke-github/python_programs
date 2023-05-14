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
from PIL import Image, ExifTags
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

class RootDirsFiles(RecordClass):
	root: str
	l_dir_name: List[str]
	l_file_name: List[str]

if __name__ == '__main__':
	# file_path_dir = os.path.join(HOME_DIR, 'Documents/picture_data_backup_sorting/main_pc_trieben')
	# file_path_dir = os.path.join(HOME_DIR, 'Documents/picture_data_backup_sorting/semi19_pc')
	file_path_dir = os.path.join(HOME_DIR, 'Documents/picture_data_backup_sorting/HZ_2TB_Intern')

	l_rootdirfile = [RootDirsFiles(*next(os.walk(file_path_dir)))]
	l_rootdirfile_temp = [rootdirfile for rootdirfile in l_rootdirfile]

	iter_nr = 0
	while len(l_rootdirfile_temp) > 0:
		iter_nr += 1
		# print("iter_nr: {}".format(iter_nr))

		len_l_rootdirfile_temp = len(l_rootdirfile_temp)
		l_rootdirfile_temp_2 = []
		for i_rootdirfile, rootdirfile in enumerate(l_rootdirfile_temp, 1):
			print(f"iter_nr: {iter_nr}, rootdirfile {i_rootdirfile:5}/{len_l_rootdirfile_temp:5}")
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

	l_file_path_abs = []

	for rootdirfile in l_rootdirfile:
		root = rootdirfile.root
		l_file_name = rootdirfile.l_file_name

		for file_name in l_file_name:
			l_file_path_abs.append(os.path.join(root, file_name))

	# sys.exit()

	l_df = []
	l_df_part = []
	for file_path in l_file_path_abs:
		print(f"file_path: {file_path}")

		with gzip.open(file_path, 'rb') as f:
			df = dill.load(f)
		
		l_df.append(df)

		# append the first column with the root_first and concat it with the df_file
		l = [
			pd.concat((
				pd.DataFrame(data=[[row['root_first']]]*row['df_file'].shape[0], columns=['root_first'], dtype=object),
				pd.DataFrame(data=[[row['rel_root']]]*row['df_file'].shape[0], columns=['rel_root'], dtype=object),
				row['df_file']
			), axis=1) for _, row in df.iterrows() if row['df_file'].shape[0] > 0
		]
		l_df_part.extend(l)

	# combine all of the df parts to a one df_file_all object
	df_file_all = pd.concat(l_df_part, axis=0)
	df_file_all.reset_index(drop=True, inplace=True)

	# add all possible extensions, include also with 1 or 2 dots too e.g. .zip or .tar.gz or .c.gz.byte etc.

	df_file_all['extension_1'] = pd.Series(data=['.'.join(name.lower().split('.')[-1:]) if name.count('.') >= 1 else '' for name in df_file_all['name']], index=df_file_all.index, dtype=object)
	df_file_all['extension_2'] = pd.Series(data=['.'.join(name.lower().split('.')[-2:]) if name.count('.') >= 2 else '' for name in df_file_all['name']], index=df_file_all.index, dtype=object)
	df_file_all['extension_3'] = pd.Series(data=['.'.join(name.lower().split('.')[-3:]) if name.count('.') >= 3 else '' for name in df_file_all['name']], index=df_file_all.index, dtype=object)

	u, c = np.unique(df_file_all['extension_1'].values, return_counts=True)
	df_extension_1 = pd.DataFrame(data={'u': u, 'c': c}, columns=['u', 'c'], dtype=object)
	df_extension_1.sort_values(by=['c', 'u'])

	u, c = np.unique(df_file_all['extension_2'].values, return_counts=True)
	df_extension_2 = pd.DataFrame(data={'u': u, 'c': c}, columns=['u', 'c'], dtype=object)

	u, c = np.unique(df_file_all['extension_3'].values, return_counts=True)
	df_extension_3 = pd.DataFrame(data={'u': u, 'c': c}, columns=['u', 'c'], dtype=object)

	u, c = np.unique(df_file_all['sha3_256sum'].values, return_counts=True)
	u2, c2 = np.unique(c, return_counts=True)

	# extract all jpg images
	df_file_jpg = df_file_all.loc[df_file_all['extension_1'].values == 'jpg'].copy()

	# if True:
	if False:
		amount_files_jpg = df_file_jpg.shape[0]
		df_file_jpg['exif'] = None
		for file_nr, (row_nr, row) in enumerate(df_file_jpg.iterrows(), 1):
			root_first = row['root_first']
			rel_root = row['rel_root']
			name = row['name']
			file_path = os.path.join(os.path.join(root_first, rel_root), name)

			print(f"file_nr: {file_nr:6}/{amount_files_jpg:6}, file_path: {file_path}")
			
			img = Image.open(file_path)
			try:
				exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
			except AttributeError as e:
				exif = None
			except:
				assert False
			del img

			row['exif'] = exif

		df_file_jpg_exif = df_file_jpg.loc[df_file_jpg['exif'].values != None].copy()
		df_file_jpg_exif.reset_index(inplace=True)

		df_file_jpg_exif['Make'] = pd.Series(data=[d['Make'] if 'Make' in d else None for d in df_file_jpg_exif['exif'].values], index=df_file_jpg_exif.index, dtype=object)
		df_file_jpg_exif['Model'] = pd.Series(data=[d['Model'] if 'Model' in d else None for d in df_file_jpg_exif['exif'].values], index=df_file_jpg_exif.index, dtype=object)
		
		df_file_jpg_exif['ExifImageWidth'] = pd.Series(data=[d['ExifImageWidth'] if 'ExifImageWidth' in d else None for d in df_file_jpg_exif['exif'].values], index=df_file_jpg_exif.index, dtype=object)
		df_file_jpg_exif['ImageWidth'] = pd.Series(data=[d['ImageWidth'] if 'ImageWidth' in d else None for d in df_file_jpg_exif['exif'].values], index=df_file_jpg_exif.index, dtype=object)
		df_file_jpg_exif['ExifImageHeight'] = pd.Series(data=[d['ExifImageHeight'] if 'ExifImageHeight' in d else None for d in df_file_jpg_exif['exif'].values], index=df_file_jpg_exif.index, dtype=object)
		df_file_jpg_exif['ImageLength'] = pd.Series(data=[d['ImageLength'] if 'ImageLength' in d else None for d in df_file_jpg_exif['exif'].values], index=df_file_jpg_exif.index, dtype=object)

		df_file_jpg_exif['width'] = pd.Series(data=[row['ExifImageWidth'] if row['ExifImageWidth'] is not None else row['ImageWidth'] if row['ImageWidth'] is not None else None for row_nr, row in df_file_jpg_exif.iterrows()], index=df_file_jpg_exif.index, dtype=object)
		df_file_jpg_exif['height'] = pd.Series(data=[row['ExifImageHeight'] if row['ExifImageHeight'] is not None else row['ImageLength'] if row['ImageLength'] is not None else None for row_nr, row in df_file_jpg_exif.iterrows()], index=df_file_jpg_exif.index, dtype=object)

		df_grpby_model = df_file_jpg_exif.groupby(by=['Make', 'Model']).apply(dict).reset_index().rename(columns={0: 'dict'})
		df_grpby_dims = df_file_jpg_exif.groupby(by=['width', 'height']).apply(dict).reset_index().rename(columns={0: 'dict'})

		df_grpby_model['amount'] = pd.Series(data=[len(row['dict']['index']) for row_nr, row in df_grpby_model.iterrows()], index=df_grpby_model.index, dtype=object)
		df_grpby_dims['amount'] = pd.Series(data=[len(row['dict']['index']) for row_nr, row in df_grpby_dims.iterrows()], index=df_grpby_dims.index, dtype=object)

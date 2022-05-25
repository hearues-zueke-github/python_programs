#! /usr/bin/python3.10

# pip installed libraries
import dill
import glob
import gzip
import keyboard
import os
import requests
import sh
import string
import subprocess
import sys
import time
import traceback
import tty
import yaml

import multiprocessing as mp
import numpy as np
import pandas as pd

from copy import deepcopy
from multiprocessing import Pool

from io import StringIO
from memory_tempfile import MemoryTempfile

from numpy.random import Generator, PCG64
from scipy.sparse import csr_matrix, coo_matrix

from PIL import Image
from typing import Dict

from neuronal_network import NeuralNetwork

HOME_DIR = os.path.expanduser("~")
PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = MemoryTempfile().gettempdir()
PYTHON_PROGRAMS_DIR = os.path.join(HOME_DIR, 'git/python_programs')

# set the relative/absolute path where the utils_load_module.py file is placed!
sys.path.append(PYTHON_PROGRAMS_DIR)
from utils_load_module import load_module_dynamically

var_glob = globals()
load_module_dynamically(**dict(var_glob=var_glob, name='utils', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils.py")))
# load_module_dynamically(**dict(var_glob=var_glob, name='utils_multiprocessing_manager', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils_multiprocessing_manager.py")))
load_module_dynamically(**dict(var_glob=var_glob, name='utils_serialization', path=os.path.join(PYTHON_PROGRAMS_DIR, "utils_serialization.py")))

mkdirs = utils.mkdirs
# MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

get_pkl_gz_obj = utils_serialization.get_pkl_gz_obj
save_pkl_gz_obj = utils_serialization.save_pkl_gz_obj
load_pkl_gz_obj = utils_serialization.load_pkl_gz_obj

save_pkl_obj = utils_serialization.save_pkl_obj
load_pkl_obj = utils_serialization.load_pkl_obj

WORKER_SLEEP_TIME = 0.001
MANAGER_SLEEP_TIME = 0.001

class MultiprocessingParallelManager():

	def __init__(self, cpu_amount):
		self.cpu_amount = cpu_amount
		assert(self.cpu_amount > 1)
		self.worker_amount = self.cpu_amount - 1

		self.l_pipe_parent_child = [mp.Pipe(duplex=False) for _ in range(0, self.cpu_amount)]
		self.l_pipe_child_parent = [mp.Pipe(duplex=False) for _ in range(0, self.cpu_amount)]

		self.l_pipe_parent_send_recv = []
		self.l_pipe_child_send_recv = []

		self.l_pipe_parent_send = []
		self.l_pipe_parent_recv = []
		
		for (pipe_child_recv, pipe_parent_send), (pipe_parent_recv, pipe_child_send) in zip(self.l_pipe_parent_child, self.l_pipe_child_parent):
			self.l_pipe_parent_send_recv.append((pipe_parent_send, pipe_parent_recv))
			self.l_pipe_child_send_recv.append((pipe_child_send, pipe_child_recv))

			self.l_pipe_parent_send.append(pipe_parent_send)
			self.l_pipe_parent_recv.append(pipe_parent_recv)

		self.l_worker_proc = [mp.Process(target=MultiprocessingParallelManager.worker_function, args=(pipe_send, pipe_recv)) for pipe_send, pipe_recv in self.l_pipe_child_send_recv]

		for proc in self.l_worker_proc:
			proc.start()

		self.current_worker_proc_idx = 0


	def __del__(self):
		for pipe_send in self.l_pipe_parent_send:
			pipe_send.send(('exit', {}))
		for proc in self.l_worker_proc:
			proc.join()


	@staticmethod
	def worker_function(pipe_send, pipe_recv):
		worker_id = None
		iter_class = None
		iter_obj = None

		while True:
			while not pipe_recv.poll():
				time.sleep(WORKER_SLEEP_TIME)
			
			recv = pipe_recv.recv()
			if not isinstance(recv, tuple):
				pipe_send.send(("Wrong type!", {}))
				continue
			elif not len(recv) == 2:
				# print(f"worker_id: {worker_id}, recv: {recv}, len(recv): {len(recv)}")
				pipe_send.send(("Wrong tuple length!", {}))
				continue

			modus, d = recv
			if not isinstance(d, dict):
				pipe_send.send((f"Wrong type of d! Needed 'dict', got '{type(d)}'!", {}))
				continue

			if modus == "init":
				if not 'iter_class' in d:
					pipe_send.send(("Needed 'iter_class' in d!", {}))
					continue
				if not 'kwargs' in d:
					pipe_send.send(("Needed 'kwargs' in d!", {}))
					continue
				if not 'worker_id' in d:
					pipe_send.send(("Needed 'worker_id' in d!", {}))
					continue

				iter_class = d['iter_class']
				kwargs = d['kwargs']
				worker_id = d['worker_id']
				try:
					iter_obj = iter_class(**kwargs)
					iter_obj.init()
				except:
					iter_class = None
					iter_obj = None
					exec_stack = traceback.format_exc()
					pipe_send.send(("Could not create 'iter_obj'!", {'exec_stack': exec_stack}))
					continue
				pipe_send.send(("Success", {}))
			elif modus == "next":
				if iter_obj is None:
					pipe_send.send(("Object 'iter_obj' was not initialized!", {}))
					continue

				if not 'args' in d:
					pipe_send.send(("Needed 'args' in d!", {}))
					continue

				args = d['args']
				try:
					ret_val = iter_obj.next(*args)
				except:
					exec_stack = traceback.format_exc()
					pipe_send.send(("Could not execute 'iter_obj.next(*args)'!", {'exec_stack': exec_stack}))
					continue
				pipe_send.send(("Success", {'ret_val': ret_val}))
			elif modus == "update":
				if iter_obj is None:
					pipe_send.send(("Object 'iter_obj' was not initialized!", {}))
					continue

				if not 'args' in d:
					pipe_send.send(("Needed 'args' in d!", {}))
					continue

				args = d['args']
				try:
					iter_obj.update(*args)
				except:
					pipe_send.send(("Could not execute 'iter_obj.update(*args)'!", {}))
					continue
				pipe_send.send(("Success", {}))
			elif modus == "save":
				if iter_obj is None:
					pipe_send.send(("Object 'iter_obj' was not initialized!", {}))
					continue

				if not 'args' in d:
					pipe_send.send(("Needed 'args' in d!", {}))
					continue

				args = d['args']
				try:
					iter_obj.save(*args)
				except:
					exec_stack = traceback.format_exc()
					# exec_stack = f"{sys.exc_info()[2]}"
					pipe_send.send(("Could not execute 'iter_obj.save(*args)'!", {'exec_stack': exec_stack}))
					continue
				pipe_send.send(("Success", {}))
			elif modus == "exit":
				break
			else:
				pipe_send.send(("Modus Is Not Defined!", {}))
				continue
		
		pipe_send.send(("Success", {}))


	def init(self, iter_class, l_kwargs, accumulate_class, kwargs_accumulate):
		assert(len(l_kwargs) == self.worker_amount)
		self.iter_class = iter_class
		self.accumulate_class = accumulate_class

		is_all_ok = True
		for worker_id, ((pipe_send, pipe_recv), kwargs) in enumerate(zip(self.l_pipe_parent_send_recv, l_kwargs), 0):
			pipe_send.send(('init', {'iter_class': iter_class, 'kwargs': kwargs, 'worker_id': worker_id}))
			recv = pipe_recv.recv()
			msg, ret_val = recv
			if "Success" != msg:
				exec_stack = ret_val['exec_stack']
				print(f"Error for worker_id '{worker_id}' with the message: '{msg}', except_stack: {exec_stack}")
				is_all_ok = False
		assert is_all_ok

		self.accumulate_obj = self.accumulate_class(**kwargs_accumulate)
		assert self.accumulate_obj.max_amount <= self.worker_amount


	def next(self, l_args_next, args_acc_next):
		max_iters = len(l_args_next)
		
		# max_iters_no_new_work = max_iters - self.worker_amount

		iters_args = 0
		l_proc_idx_free = list(range(0, self.worker_amount))
		l_proc_idx_working = []
		# if max_iters >= self.worker_amount:
		for _ in range(0, min(max_iters, self.worker_amount)):
			worker_id = l_proc_idx_free.pop(0)
			l_proc_idx_working.append(worker_id)

			self.l_pipe_parent_send[worker_id].send(('next', {'args': l_args_next[iters_args]}))
			iters_args += 1
		# else:
		# 	assert False and "Not Implemented yet!"

		iters_args_finished = 0
		while iters_args_finished < max_iters:
			next_amount = self.accumulate_obj.get_next_amount()

			iters_next = iters_args_finished + next_amount
			if iters_next > max_iters:
				iters_next = max_iters
				next_amount = iters_next - iters_args_finished

			l_proc_idx_done = []
			self.accumulate_obj.init()
			for _ in range(0, next_amount):
				while True:
					worker_id = l_proc_idx_working.pop(0)
					if not self.l_pipe_parent_recv[worker_id].poll():
						l_proc_idx_working.append(worker_id)
						time.sleep(MANAGER_SLEEP_TIME)
						continue
					break

				l_proc_idx_done.append(worker_id)

				recv = self.l_pipe_parent_recv[worker_id].recv()
				msg, d = recv
				# print(f"msg: {msg}")
				try:
					assert msg == "Success"
				except:
					exec_stack = d['exec_stack']
					print(f"Error for worker_id '{worker_id}' with the message: '{msg}', except_stack: {exec_stack}")
					assert False
				ret_val = d['ret_val']

				self.accumulate_obj.accumulate(*ret_val)

			self.accumulate_obj.update()

			for worker_id in l_proc_idx_done:
				args = (self.accumulate_obj.next(*args_acc_next), )
				self.l_pipe_parent_send[worker_id].send(('update', {'args': args}))

			print(f"print_prefix: [{self.accumulate_obj.print_prefix}], iters_args_finished: {iters_args_finished}, accumulate_obj.current():\n{self.accumulate_obj.current()}")

			for worker_id in l_proc_idx_done:
				recv = self.l_pipe_parent_recv[worker_id].recv()
				msg, ret_val = recv
				assert msg == "Success"

			for worker_id in l_proc_idx_done:
				l_proc_idx_free.append(worker_id)

				if iters_args < max_iters:
					l_proc_idx_free.pop(l_proc_idx_free.index(worker_id))
					l_proc_idx_working.append(worker_id)
							
					self.l_pipe_parent_send[worker_id].send(('next', {'args': l_args_next[iters_args]}))
					
					iters_args += 1

			iters_args_finished = iters_next

		assert len(l_proc_idx_free) == self.worker_amount
		assert len(l_proc_idx_working) == 0

		return self.accumulate_obj.current()

	def save(self, l_args_iter_save):
		assert len(l_args_iter_save) == self.worker_amount

		for worker_id in range(0, self.worker_amount):
			self.l_pipe_parent_send[worker_id].send(('save', {'args': l_args_iter_save[worker_id]}))

		for worker_id in range(0, self.worker_amount):
			recv = self.l_pipe_parent_recv[worker_id].recv()
			msg, ret_val = recv
			try:
				assert msg == 'Success'
			except:
				exec_stack = ret_val['exec_stack']
				print(f"exec_stack:\n{exec_stack}")
				assert False


if __name__ == '__main__':
	print("Hello World!")
	# TODO: make a very simple example for demonstration only

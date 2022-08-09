import dill
import os
import sys
import time
import traceback

from collections import deque
from multiprocessing import Process, Pipe

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if PATH_ROOT_DIR not in sys.path:
	sys.path.append(PATH_ROOT_DIR)

'''
Example of usage:

def f(x):
	return x**2

mult_proc_mng = MultiprocessingManager(cpu_count=mp.cpu_count())

# # only for testing the responsivness!
# mult_proc_mng.test_worker_threads_response()

print('Define new Function!')
mult_proc_mng.define_new_func('func_f', f)
print('Do the jobs!!')
l_arguments = [(x*2, ) for x in range(0, 100)]
l_ret = mult_proc_mng.do_new_jobs(
	['func_f']*len(l_arguments),
	l_arguments,
)
print("len(l_ret): {}".format(len(l_ret)))
# print("l_ret: {}".format(l_ret))

# # testing the responsivness again!
# mult_proc_mng.test_worker_threads_response()
del mult_proc_mng
'''

WORKER_SLEEP_TIME = 0.001
MANAGER_SLEEP_TIME = 0.001

class MultiprocessingManager(Exception):
	def __init__(self, cpu_count, is_print_on=True):
		self.cpu_count = cpu_count
		self.worker_amount = self.cpu_count - 1
		self.is_print_on = is_print_on

		# 1 proc for manager (class itself)
		# cpu_count-1 procs for the worker threads (processes)

		self.pipes_recv_worker, self.pipes_send_main = list(zip(*[Pipe(duplex=False) for _ in range(0, self.worker_amount)]))
		self.pipes_recv_main, self.pipes_send_worker = list(zip(*[Pipe(duplex=False) for _ in range(0, self.worker_amount)]))

		self.l_worker_proc = [
			Process(target=self._worker_thread, args=(i, pipe_in, pipe_out))
			for i, pipe_in, pipe_out in zip(range(0, self.worker_amount), self.pipes_recv_worker, self.pipes_send_worker)
		]
		for proc in self.l_worker_proc:
			proc.start()


	def _worker_thread(self, worker_nr, pipe_in, pipe_out):
		d_func = {}
		while True:
			if pipe_in.poll():
				(name, args) = pipe_in.recv()
				if name == 'exit':
					break
				elif name == 'func_def_new':
					func_name, func_bytes = args
					d_func[func_name] = dill.loads(func_bytes)
					pipe_out.send((worker_nr, "Finished 'func_def_new'"))
				elif name == 'func_def_exec':
					func_name, func_args = args
					try:
						ret_val = d_func[func_name](*func_args)
					except:
						if self.is_print_on:
							print('Fail for func_name: "{}", func_args: "{}", at worker_nr: {}'.format(func_name, [str(arg)[:50] for arg in func_args], worker_nr))
							traceback.print_stack()

							_, _, tb = sys.exc_info()
							print('worker_nr: {}, traceback.format_list:\n{}'.format(worker_nr, ''.join(traceback.format_list(traceback.extract_tb(tb)))))
						ret_val = None
					ret_tpl = (worker_nr, ret_val)
					pipe_out.send(ret_tpl)
				elif name == 'test_ret':
					pipe_out.send('IS WORKING!')
			time.sleep(WORKER_SLEEP_TIME)


	def define_new_func(self, name, func):
		func_bytes = dill.dumps(func)
		for pipe_send in self.pipes_send_main:
			pipe_send.send(('func_def_new', (name, func_bytes)))

		for pipe_recv in self.pipes_recv_main:
			ret = pipe_recv.recv()
			worker_nr, text = ret
			if self.is_print_on:
				print("worker_nr: {}, text: {}".format(worker_nr, text))


	def test_worker_threads_response(self):
		for pipe_send in self.pipes_send_main:
			pipe_send.send(('test_ret', ()))

		for pipe_recv in self.pipes_recv_main:
			ret = pipe_recv.recv()
			assert ret == 'IS WORKING!'
		if self.is_print_on:
			pass
			# print('IS WORKING OK!!!')


	def do_new_jobs(self, l_func_name, l_func_args):
		len_l_func_name = len(l_func_name)

		l_ret = []
		if len_l_func_name <= self.worker_amount:
			for worker_nr, (pipe_send, func_name, func_args) in enumerate(zip(self.pipes_send_main, l_func_name, l_func_args), 0):
				pipe_send.send(('func_def_exec', (func_name, func_args)))
				if self.is_print_on:
					print("Doing job: worker_nr: {}".format(worker_nr))

			finished_works = 0
			dq_pipe_i = deque(range(0, len_l_func_name))
			while len(dq_pipe_i) > 0:
				pipe_i = dq_pipe_i.popleft()

				pipe_recv = self.pipes_recv_main[pipe_i]
				if not pipe_recv.poll():
					dq_pipe_i.append(pipe_i)
					time.sleep(MANAGER_SLEEP_TIME)
					continue

				ret_tpl = pipe_recv.recv()
				worker_nr, ret_val = ret_tpl
				l_ret.append(ret_val)
				
				finished_works += 1
				if self.is_print_on:
					print("Finished: {:2}/{:2}, worker_nr: {}".format(finished_works, len_l_func_name, worker_nr))
		else:
			for worker_nr, (pipe_send, func_name, func_args) in enumerate(zip(self.pipes_send_main, l_func_name[:self.worker_amount], l_func_args[:self.worker_amount]), 0):
				pipe_send.send(('func_def_exec', (func_name, func_args)))
				if self.is_print_on:
					print("Doing job: worker_nr: {}".format(worker_nr))
			
			finished_works = 0
			pipe_i = 0
			for func_name, func_args in zip(l_func_name[self.worker_amount:], l_func_args[self.worker_amount:]):
				while True:
					pipe_recv = self.pipes_recv_main[pipe_i]
					if not pipe_recv.poll():
						time.sleep(MANAGER_SLEEP_TIME)
						pipe_i = (pipe_i+1) % self.worker_amount
						continue
					break

				ret_tpl = pipe_recv.recv()
				worker_nr, ret_val = ret_tpl
				l_ret.append(ret_val)

				finished_works += 1
				if self.is_print_on:
					print("Finished: {:2}/{:2}, worker_nr: {}".format(finished_works, len_l_func_name, worker_nr))

				pipe_send = self.pipes_send_main[pipe_i]
				pipe_send.send(('func_def_exec', (func_name, func_args)))

				if self.is_print_on:
					print("Doing job: worker_nr: {}".format(worker_nr))
				
				pipe_i = (pipe_i+1) % self.worker_amount
			
			dq_pipe_i = deque(range(0, self.worker_amount))
			while len(dq_pipe_i) > 0:
				pipe_i = dq_pipe_i.popleft()

				pipe_recv = self.pipes_recv_main[pipe_i]
				if not pipe_recv.poll():
					dq_pipe_i.append(pipe_i)
					time.sleep(MANAGER_SLEEP_TIME)
					continue

				ret_tpl = pipe_recv.recv()
				worker_nr, ret_val = ret_tpl
				l_ret.append(ret_val)
				
				finished_works += 1
				if self.is_print_on:
					print("Finished: {:2}/{:2}, worker_nr: {}".format(finished_works, len_l_func_name, worker_nr))

		return l_ret


	def __del__(self):
		for pipe_send in self.pipes_send_main:
			pipe_send.send(('exit', ()))
		for proc in self.l_worker_proc:
			proc.join()

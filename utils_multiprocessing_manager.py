import dill
import time

from collections import deque
from multiprocessing import Process, Pipe

WORKER_SLEEP_TIME = 0.02
MANAGER_SLEEP_TIME = 0.02

class MultiprocessingManager(Exception):
    def __init__(self, cpu_count):
        self.cpu_count = cpu_count
        self.worker_amount = self.cpu_count - 1

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
                        print('Fail for func_name: {}, func_args: {}, at worker_nr: {}'.format(func_name, func_args, worker_nr))
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
            print("worker_nr: {}, text: {}".format(worker_nr, text))


    def test_worker_threads_response(self):
        for pipe_send in self.pipes_send_main:
            pipe_send.send(('test_ret', ()))

        for pipe_recv in self.pipes_recv_main:
            ret = pipe_recv.recv()
            assert ret == 'IS WORKING!'
        # print('IS WORKING OK!!!')


    def do_new_jobs(self, l_func_name, l_func_args):
        len_l_func_name = len(l_func_name)

        l_ret = []
        if len_l_func_name <= self.worker_amount:
            for worker_nr, (pipe_send, func_name, func_args) in enumerate(zip(self.pipes_send_main, l_func_name, l_func_args), 0):
                pipe_send.send(('func_def_exec', (func_name, func_args)))
                print("Doing job: worker_nr: {}".format(worker_nr))

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
                
                print("Finished: worker_nr: {}".format(worker_nr))
        else:
            for worker_nr, (pipe_send, func_name, func_args) in enumerate(zip(self.pipes_send_main, l_func_name[:self.worker_amount], l_func_args[:self.worker_amount]), 0):
                pipe_send.send(('func_def_exec', (func_name, func_args)))
                print("Doing job: worker_nr: {}".format(worker_nr))
            
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

                print("Finished: worker_nr: {}".format(worker_nr))

                pipe_send = self.pipes_send_main[pipe_i]
                pipe_send.send(('func_def_exec', (func_name, func_args)))

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
                
                print("Finished: worker_nr: {}".format(worker_nr))

        return l_ret


    def __del__(self):
        for pipe_send in self.pipes_send_main:
            pipe_send.send(('exit', ()))
        for proc in self.l_worker_proc:
            proc.join()

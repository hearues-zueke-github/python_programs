#! /usr/bin/env -S /usr/bin/time /usr/bin/python3.9.5 -i

# -*- coding: utf-8 -*-

# Some other needed imports
import os
import sys

from memory_tempfile import MemoryTempfile
from typing import List, Set, Tuple, Dict, Union

from inspect import currentframe, getframeinfo

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("~")
TEMP_DIR = MemoryTempfile().gettempdir()

import importlib.util as imp_util

# TODO: change the optaining of the git root folder from a config file first!
spec = imp_util.spec_from_file_location("utils", os.path.join(HOME_DIR, "git/python_programs/utils.py"))
utils = imp_util.module_from_spec(spec)
spec.loader.exec_module(utils)

mkdirs = utils.mkdirs

spec = imp_util.spec_from_file_location("utils_multiprocessing_manager", os.path.join(HOME_DIR, "git/python_programs/utils_multiprocessing_manager.py"))
utils_multiprocessing_manager = imp_util.module_from_spec(spec)
spec.loader.exec_module(utils_multiprocessing_manager)

MultiprocessingManager = utils_multiprocessing_manager.MultiprocessingManager

OBJS_DIR_PATH = PATH_ROOT_DIR+'objs'
mkdirs(OBJS_DIR_PATH)

# create dynamically the BYTE_ variables

l_char = list('gbnhe0123456789ABCDEF')
d_char_to_int = {}
for c in l_char:
    d_char_to_int[c] = list(bytes(c.encode(encoding='utf-8')))[0]

for c, value in d_char_to_int.items():
    globals()[f'BYTE_{c}'] = value

l_char_operator = [
    ('add', '+'),
    ('sub', '-'),
    ('mul', '*'),
    ('div', '/'),
    ('mod', '%'),
]
for name, c in l_char_operator: 
    value = list(bytes(c.encode(encoding='utf-8')))[0]
    d_char_to_int[c] = value
    globals()[f'BYTE_{name}'] = value

# all possible values!
s_char_byte = set(list(d_char_to_int.values())+[b'\x00'])

SYMBOLS_BIN: Set[int] = set(list(b'01'))
SYMBOLS_DEC: Set[int] = set(list(b'0123456789'))
SYMBOLS_HEX: Set[int] = set(list(b'0123456789ABCDEF'))

class OneByteOpcodeLang(Exception):
    __slots__ = ['l_memory', 'l_stack', 'mem_pos']

    def __init__(self):
        self.empty_all_vars()

    def empty_all_vars(self) -> None:
        self.l_memory = []
        self.l_stack = []
        self.mem_pos = 0

    def parse_interpret_str_input(self, input_str: str) -> None:
        self.empty_all_vars()

        self.l_memory = list(input_str)
        self.mem_pos = 0
        self.f_main()

    def f_main(self) -> None:
        frameinfo = getframeinfo(currentframe())
        print("filename: {}, fileno: {}".format(frameinfo.filename, frameinfo.lineno))

        print("f_main: self.mem_pos: {}".format(self.mem_pos))
        c = self.l_memory[self.mem_pos]
        while True:
            if c == BYTE_g:
                self.l_stack.append(BYTE_g)
                self.mem_pos += 1
                self.f_assign()
            elif c == BYTE_add:
                self.l_stack.append(BYTE_add)
                self.mem_pos += 1
                self.f_add()
            elif c == 0:
                break
            else:
                assert False

            if len(self.l_memory) <= self.mem_pos:
                self.l_memory.append(0)

            c = self.l_memory[self.mem_pos]

        print("self.l_memory: {}".format(self.l_memory))
        print("self.l_stack: {}".format(self.l_stack))
        print("self.mem_pos: {}".format(self.mem_pos))
        print()

    def f_assign(self) -> None:
        frameinfo = getframeinfo(currentframe())
        print("filename: {}, fileno: {}".format(frameinfo.filename, frameinfo.lineno))

        print("f_assign: self.mem_pos: {}".format(self.mem_pos))
        c = self.l_memory[self.mem_pos]
        while True:
            if c == BYTE_b:
                self.l_stack.append(BYTE_b)
                self.mem_pos += 1
                self.f_num_bin()
            elif c == BYTE_n:
                self.l_stack.append(BYTE_n)
                self.mem_pos += 1
                self.f_num_dec()
            elif c == BYTE_h:
                self.l_stack.append(BYTE_h)
                self.mem_pos += 1
                self.f_num_hex()
            elif c == BYTE_add:
                self.l_stack.append(BYTE_add)
                self.mem_pos += 1
                self.f_add()
            elif c == BYTE_e:
                # interpret the stack now!
                i = len(self.l_stack)
                while self.l_stack[i - 1] != BYTE_g:
                    i -= 1
                ptr_pos = self.l_stack[i]
                val = self.l_stack[i + 1]
                self.l_stack = self.l_stack[:i - 1]
                while len(self.l_memory) <= ptr_pos:
                    self.l_memory.append(0)
                self.l_memory[ptr_pos] = val

                self.mem_pos += 1
                break
            else:
                assert False

            c = self.l_memory[self.mem_pos]

    def f_add(self) -> None:
        frameinfo = getframeinfo(currentframe())
        print("filename: {}, fileno: {}".format(frameinfo.filename, frameinfo.lineno))

        print("f_add: self.mem_pos: {}".format(self.mem_pos))
        c = self.l_memory[self.mem_pos]
        while True:
            if c == BYTE_b:
                self.l_stack.append(BYTE_b)
                self.mem_pos += 1
                self.f_num_bin()
            elif c == BYTE_n:
                self.l_stack.append(BYTE_n)
                self.mem_pos += 1
                self.f_num_dec()
            elif c == BYTE_h:
                self.l_stack.append(BYTE_h)
                self.mem_pos += 1
                self.f_num_hex()
            elif c == BYTE_add:
                self.l_stack.append(BYTE_add)
                self.mem_pos += 1
                self.f_add()
            elif c == BYTE_e:
                # interpret the stack now!
                i = len(self.l_stack)
                while self.l_stack[i - 1] != BYTE_add:
                    i -= 1
                val1 = self.l_stack[i]
                val2 = self.l_stack[i + 1]
                self.l_stack = self.l_stack[:i]
                self.l_stack[i - 1] = val1 + val2

                self.mem_pos += 1
                break
            else:
                assert False

            c = self.l_memory[self.mem_pos]

    def f_num_bin(self) -> None:
        frameinfo = getframeinfo(currentframe())
        print("filename: {}, fileno: {}".format(frameinfo.filename, frameinfo.lineno))

        print("f_num_bin: self.mem_pos: {}".format(self.mem_pos))
        c = self.l_memory[self.mem_pos]
        while c in SYMBOLS_BIN:
            self.l_stack.append(c)
            self.mem_pos += 1
            c = self.l_memory[self.mem_pos]
        assert c in s_char_byte

        # interpret the number on the stack!
        i = len(self.l_stack)
        while self.l_stack[i - 1] != BYTE_b:
            i -= 1
        num = int(''.join(bytes(self.l_stack[i:]).decode('utf-8')), base=2)
        self.l_stack = self.l_stack[:i]
        self.l_stack[-1] = num

    def f_num_dec(self) -> None:
        frameinfo = getframeinfo(currentframe())
        print("filename: {}, fileno: {}".format(frameinfo.filename, frameinfo.lineno))

        print("f_num_dec: self.mem_pos: {}".format(self.mem_pos))
        c = self.l_memory[self.mem_pos]
        while c in SYMBOLS_DEC:
            self.l_stack.append(c)
            self.mem_pos += 1
            c = self.l_memory[self.mem_pos]
        assert c in s_char_byte

        # interpret the number on the stack!
        i = len(self.l_stack)
        while self.l_stack[i - 1] != BYTE_n:
            i -= 1
        num = int(''.join(bytes(self.l_stack[i:]).decode('utf-8')), base=10)
        self.l_stack = self.l_stack[:i]
        self.l_stack[-1] = num

    def f_num_hex(self) -> None:
        frameinfo = getframeinfo(currentframe())
        print("filename: {}, fileno: {}".format(frameinfo.filename, frameinfo.lineno))
        
        print("f_num_hex: self.mem_pos: {}".format(self.mem_pos))
        c = self.l_memory[self.mem_pos]
        while c in SYMBOLS_HEX:
            self.l_stack.append(c)
            self.mem_pos += 1
            c = self.l_memory[self.mem_pos]
        assert c in s_char_byte

        # interpret the number on the stack!
        i = len(self.l_stack)
        while self.l_stack[i - 1] != BYTE_h:
            i -= 1
        num = int(''.join(bytes(self.l_stack[i:]).decode('utf-8')), base=16)
        self.l_stack = self.l_stack[:i]
        self.l_stack[-1] = num


class OneByteOpcodeLangSimple(Exception):
    __slots__ = ['l_memory', 'l_stack', 'mem_pos']

    def __init__(self):
        self.empty_all_vars()

    def empty_all_vars(self) -> None:
        self.l_memory = []
        self.l_stack = []
        self.mem_pos = 0

    def parse_interpret_str_input(self, input_str: str) -> None:
        self.empty_all_vars()

        self.l_memory = list(input_str)
        self.mem_pos = 0
        self.f_main()

    def f_main(self) -> None:
        frameinfo = getframeinfo(currentframe())
        print("filename: {}, fileno: {}".format(frameinfo.filename, frameinfo.lineno))

        print("f_main: self.mem_pos: {}".format(self.mem_pos))
        c = self.l_memory[self.mem_pos]
        while True:
            if c == BYTE_n or c == BYTE_h or c == BYTE_b:
                self.f_assign()
            elif c == 0:
                break
            else:
                assert False

            if len(self.l_memory) <= self.mem_pos:
                self.l_memory.append(0)

            c = self.l_memory[self.mem_pos]

        print("self.l_memory: {}".format(self.l_memory))
        print("self.l_stack: {}".format(self.l_stack))
        print("self.mem_pos: {}".format(self.mem_pos))
        print()

    def f_assign(self) -> None:
        frameinfo = getframeinfo(currentframe())
        print("filename: {}, fileno: {}".format(frameinfo.filename, frameinfo.lineno))

        print("f_assign: self.mem_pos: {}".format(self.mem_pos))
        c = self.l_memory[self.mem_pos]
        if c == BYTE_b:
            self.l_stack.append(BYTE_b)
            self.mem_pos += 1
            self.f_num_bin()
        elif c == BYTE_n:
            self.l_stack.append(BYTE_n)
            self.mem_pos += 1
            self.f_num_dec()
        elif c == BYTE_h:
            self.l_stack.append(BYTE_h)
            self.mem_pos += 1
            self.f_num_hex()
        else:
            assert False

        c = self.l_memory[self.mem_pos]
        if c == BYTE_b:
            self.l_stack.append(BYTE_b)
            self.mem_pos += 1
            self.f_num_bin()
        elif c == BYTE_n:
            self.l_stack.append(BYTE_n)
            self.mem_pos += 1
            self.f_num_dec()
        elif c == BYTE_h:
            self.l_stack.append(BYTE_h)
            self.mem_pos += 1
            self.f_num_hex()
        elif c == BYTE_add:
            self.l_stack.append(BYTE_add)
            self.mem_pos += 1
            self.f_add()
        elif c == BYTE_e:
            # interpret the stack now!
            i = len(self.l_stack)
            while self.l_stack[i - 1] != BYTE_g:
                i -= 1
            ptr_pos = self.l_stack[i]
            val = self.l_stack[i + 1]
            self.l_stack = self.l_stack[:i - 1]
            while len(self.l_memory) <= ptr_pos:
                self.l_memory.append(0)
            self.l_memory[ptr_pos] = val

            self.mem_pos += 1
            break
        else:
            assert False

        c = self.l_memory[self.mem_pos]

    def f_add(self) -> None:
        frameinfo = getframeinfo(currentframe())
        print("filename: {}, fileno: {}".format(frameinfo.filename, frameinfo.lineno))

        print("f_add: self.mem_pos: {}".format(self.mem_pos))
        c = self.l_memory[self.mem_pos]
        while True:
            if c == BYTE_b:
                self.l_stack.append(BYTE_b)
                self.mem_pos += 1
                self.f_num_bin()
            elif c == BYTE_n:
                self.l_stack.append(BYTE_n)
                self.mem_pos += 1
                self.f_num_dec()
            elif c == BYTE_h:
                self.l_stack.append(BYTE_h)
                self.mem_pos += 1
                self.f_num_hex()
            elif c == BYTE_add:
                self.l_stack.append(BYTE_add)
                self.mem_pos += 1
                self.f_add()
            elif c == BYTE_e:
                # interpret the stack now!
                i = len(self.l_stack)
                while self.l_stack[i - 1] != BYTE_add:
                    i -= 1
                val1 = self.l_stack[i]
                val2 = self.l_stack[i + 1]
                self.l_stack = self.l_stack[:i]
                self.l_stack[i - 1] = val1 + val2

                self.mem_pos += 1
                break
            else:
                assert False

            c = self.l_memory[self.mem_pos]

    def f_num_bin(self) -> None:
        frameinfo = getframeinfo(currentframe())
        print("filename: {}, fileno: {}".format(frameinfo.filename, frameinfo.lineno))

        print("f_num_bin: self.mem_pos: {}".format(self.mem_pos))
        c = self.l_memory[self.mem_pos]
        while c in SYMBOLS_BIN:
            self.l_stack.append(c)
            self.mem_pos += 1
            c = self.l_memory[self.mem_pos]
        assert c in s_char_byte

        # interpret the number on the stack!
        i = len(self.l_stack)
        while self.l_stack[i - 1] != BYTE_b:
            i -= 1
        num = int(''.join(bytes(self.l_stack[i:]).decode('utf-8')), base=2)
        self.l_stack = self.l_stack[:i]
        self.l_stack[-1] = num

    def f_num_dec(self) -> None:
        frameinfo = getframeinfo(currentframe())
        print("filename: {}, fileno: {}".format(frameinfo.filename, frameinfo.lineno))

        print("f_num_dec: self.mem_pos: {}".format(self.mem_pos))
        c = self.l_memory[self.mem_pos]
        while c in SYMBOLS_DEC:
            self.l_stack.append(c)
            self.mem_pos += 1
            c = self.l_memory[self.mem_pos]
        assert c in s_char_byte

        # interpret the number on the stack!
        i = len(self.l_stack)
        while self.l_stack[i - 1] != BYTE_n:
            i -= 1
        num = int(''.join(bytes(self.l_stack[i:]).decode('utf-8')), base=10)
        self.l_stack = self.l_stack[:i]
        self.l_stack[-1] = num

    def f_num_hex(self) -> None:
        frameinfo = getframeinfo(currentframe())
        print("filename: {}, fileno: {}".format(frameinfo.filename, frameinfo.lineno))
        
        print("f_num_hex: self.mem_pos: {}".format(self.mem_pos))
        c = self.l_memory[self.mem_pos]
        while c in SYMBOLS_HEX:
            self.l_stack.append(c)
            self.mem_pos += 1
            c = self.l_memory[self.mem_pos]
        assert c in s_char_byte

        # interpret the number on the stack!
        i = len(self.l_stack)
        while self.l_stack[i - 1] != BYTE_h:
            i -= 1
        num = int(''.join(bytes(self.l_stack[i:]).decode('utf-8')), base=16)
        self.l_stack = self.l_stack[:i]
        self.l_stack[-1] = num


if __name__ == '__main__':
    one_byte_opcode_lang = OneByteOpcodeLang()
    one_byte_opcode_lang_simple = OneByteOpcodeLangSimple()
    # code = b'gn2n8e'

    # one_byte_opcode_lang.parse_interpret_str_input(code)
    
    # assert one_byte_opcode_lang.l_memory == [103, 110, 8, 110, 56, 101, 0]
    # assert one_byte_opcode_lang.l_stack == []
    # assert one_byte_opcode_lang.mem_pos == 6

    # code = b'gn2n8n2egn3n100e'
    # one_byte_opcode_lang.parse_interpret_str_input(code)

    # code = b'ghAhBe'
    # one_byte_opcode_lang.parse_interpret_str_input(code)

    # code = b'gb100b10e'
    # one_byte_opcode_lang.parse_interpret_str_input(code)

    # sys.exit()

    # code = b'gn1+n2n3ee'
    # one_byte_opcode_lang.parse_interpret_str_input(code)

    # code = b'g+n1n0e+n2n3ee'
    # one_byte_opcode_lang.parse_interpret_str_input(code)
    
    # code = b'g+n1n0e++n0n2en3eegn0+n1+n2n3eee'
    # one_byte_opcode_lang.parse_interpret_str_input(code)

    # a much more simplified version!
    # n... decimal num, h... hex num, b... binary num
    # j... jump if the value is zero
    # p... std out, only one character
    # u... std in, user input at the specific position
    # q... pointer to the variable

    code = b'n10n4n20+n2qn5'
    one_byte_opcode_lang_simple.parse_interpret_str_input(code)

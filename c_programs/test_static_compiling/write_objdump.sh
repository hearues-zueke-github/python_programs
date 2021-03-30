#! /bin/bash

mkdir -p objdumps

objdump -d objs/main_without_simple_library.o > objdumps/objdump_main_without_simple_library.txt
objdump -d objs/main_with_simple_library.o > objdumps/objdump_main_with_simple_library.txt
objdump -d objs/simple_library.o > objdumps/objdump_simple_library.txt

objdump -d objs/main_without_simple_library_O2.o > objdumps/objdump_main_without_simple_library_O2.txt
objdump -d objs/main_with_simple_library_O2.o > objdumps/objdump_main_with_simple_library_O2.txt
objdump -d objs/simple_library_O2.o > objdumps/objdump_simple_library_O2.txt

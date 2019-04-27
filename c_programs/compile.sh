#! /bin/bash

# gcc -Wall set_bigger_stack.c -o set_bigger_stack
gcc -Wall -fsplit-stack set_bigger_stack.c -o set_bigger_stack.o
gcc -Wall increase_stack_size.c -o increase_stack_size.o

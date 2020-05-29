CC=gcc
CFLAGS=-Wall

all:
	gcc -Wall huge_number_test.c -o huge_number_test -lgmp

sequence_generator: sequence_generator.c
	$(CC) $(CFLAGS) sequence_generator.c -o sequence_generator.o

simple: simple_c_example.c
	$(CC) $(CFLAGS) simple_c_example.c -o simple_c_example.o

simple_program: simple_program.c
	$(CC) $(CFLAGS) simple_program.c -o simple_program.o

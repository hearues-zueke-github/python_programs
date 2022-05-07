.DEFAULT_GOAL := all

CC=g++
OPTIMIZATION=-O2
PREFIX=/home/doublepmcl/programs/GCC-11.1.0
LINKING=-Wl,-rpath,$(PREFIX)/lib64
CFLAGS=-Werror -Wall -std=c++20 $(OPTIMIZATION) -pthread $(LINKING)
# CFLAGS=-Werror -Wall -std=c++20 $(OPTIMIZATION) -pthread -fmodules-ts $(LINKING)

iostream:
	$(CC) $(CFLAGS) -x c++-system-header iostream -c -o iostream.o
system_modules:
	$(CC) $(CFLAGS) -x c++-system-header iostream vector -c
utils.o: utils.cpp utils.h
	$(CC) $(CFLAGS) utils.cpp -c
thread_data_generic.o: thread_data_generic.cpp thread_data_generic.h
	$(CC) $(CFLAGS) thread_data_generic.cpp -c
multi_linear_sequences.o: multi_linear_sequences.cpp multi_linear_sequences.h
	$(CC) $(CFLAGS) multi_linear_sequences.cpp -c
main.o: utils.o thread_data_generic.o multi_linear_sequences.o main.cpp
	$(CC) $(CFLAGS) $(LINKING) utils.o thread_data_generic.o multi_linear_sequences.o main.cpp -o main.o
# 	$(CC) $(CFLAGS) $(LINKING) utils.o thread_data_generic.o multi_linear_sequences.o gcm.cache/iostream.gcm main.cpp -o main.o
all: main.o

clean:
	rm utils.o
	rm thread_data_generic.o
	rm main.o

.DEFAULT_GOAL := all

CC=g++
# CFLAGS=-Werror -Wall -std=c++20 -g -pthread
CFLAGS=-Werror -Wall -std=c++20 -O2 -pthread
# CFLAGS=-Wall -std=c++20

utils: utils.cpp
	$(CC) $(CFLAGS) utils.cpp -c
main: main.cpp utils.o
	$(CC) $(CFLAGS) main.cpp utils.o -o main.o
all: main

clean:
	rm utils.o
	rm main.o

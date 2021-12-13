.DEFAULT_GOAL := all

CC=g++
CFLAGS=-Werror -Wall -std=c++20 -g
# CFLAGS=-Werror -Wall -std=c++20 -O2 -g
# CFLAGS=-Wall -std=c++20

utils: utils.cpp
	$(CC) $(CFLAGS) utils.cpp -c
main: main.cpp utils
	$(CC) $(CFLAGS) main.cpp utils.o -o main.o
all: main

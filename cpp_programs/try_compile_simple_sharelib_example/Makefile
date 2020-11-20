.PHONY: all
CC=g++
CFLAGS=-Wall -std=c++17 -O2

utils:
	$(CC) $(CFLAGS) utils.cpp -c
all: utils
	$(CC) $(CFLAGS) main.cpp utils.o -o main.o

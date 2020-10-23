.PHONY: default
CC=g++
CFLAGS=-Wall -std=c++17

Numbers:
	$(CC) $(CFLAGS) Numbers.cpp -c -o Numbers.o
Factors:
	$(CC) $(CFLAGS) Factors.cpp -c -o Factors.o
Polynome: Factors
	$(CC) $(CFLAGS) Polynome.cpp -c -o Polynome.o
Polynomial: Polynome
	$(CC) $(CFLAGS) Polynomial.cpp -c -o Polynomial.o
default: Numbers Polynome Polynomial
	$(CC) $(CFLAGS) main.cpp Numbers.o Polynome.o Polynomial.o -o main.o

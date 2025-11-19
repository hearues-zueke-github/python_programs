HEADERS = cyclic_multi_factor_pow_seq.h

default: cyclic_multi_factor_pow_seq

cyclic_multi_factor_pow_seq.o: cyclic_multi_factor_pow_seq.cpp $(HEADERS)
	g++ -c cyclic_multi_factor_pow_seq.cpp -std=c++23 -lsqlite3 -O3 -o cyclic_multi_factor_pow_seq.o

cyclic_multi_factor_pow_seq: cyclic_multi_factor_pow_seq.o
	g++ cyclic_multi_factor_pow_seq.o -std=c++23 -lsqlite3 -O3 -o cyclic_multi_factor_pow_seq

clean:
	-rm -f cyclic_multi_factor_pow_seq.o
	-rm -f cyclic_multi_factor_pow_seq

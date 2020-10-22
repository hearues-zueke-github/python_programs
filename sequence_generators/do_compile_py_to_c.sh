#! /bin/bash

PYTHON_PATH="/home/doublepmcl/programs/Python-3.8.6"

mkdir -p build
# cython -3 test.py -o build/sequence_gen_1.c
python3.6 -m cython -3 test.py -o build/sequence_gen_1.c --embed
# cython -3 test.py -o build/sequence_gen_1.c --embed
# cython -3 sequence_gen_1.py -o build/sequence_gen_1.c --embed
cd build
# gcc sequence_gen_1.c -o sequence_gen_1.o -I$PYTHON_PATH -I$PYTHON_PATH/Include -L$PYTHON_PATH -L$PYTHON_PATH/libpython3.8.so.1.0 -L$PYTHON_PATH/libpython3.8.so  -L$PYTHON_PATH/libpython3.8 -L$PYTHON_PATH/build/lib.linux-x86_64-3.8

# gcc \
# -Os \
# -I/usr/include/python3.6m \
# -I/usr/include/python3.6m/Include \
# sequence_gen_1.c \
# -lpython3.6m \
# -o sequence_gen_1.o \
# -lpthread -lm -lutil -ldl

gcc \
-Os \
-I$PYTHON_PATH \
-I$PYTHON_PATH/Include \
sequence_gen_1.c \
-L$PYTHON_PATH/libpython3.8.so \
-o sequence_gen_1.o \
-lpthread -lm -lutil -ldl

# -L$PYTHON_PATH/libpython3.so \

# gcc -fPIC -shared sequence_gen_1.c -o sequence_gen_1.o -I/usr/local/include/python3.8 -I/usr/local/include/python3.8/Include -L$PYTHON_PATH/libpython3.so
# gcc -fPIC sequence_gen_1.c -o sequence_gen_1.o -I/usr/local/include/python3.8 -I/usr/local/include/python3.8/Include -L$PYTHON_PATH/libpython3.so
# gcc sequence_gen_1.c -o sequence_gen_1.o -I/usr/local/include/python3.8 -I/usr/local/include/python3.8/Include # -I$PYTHON_PATH -I$PYTHON_PATH/Include -L$PYTHON_PATH -L$PYTHON_PATH/libpython3.8.so.1.0


# gcc sequence_gen_1.c -o sequence_gen_1.o # -I$PYTHON_PATH -I$PYTHON_PATH/Include -L$PYTHON_PATH -L$PYTHON_PATH/libpython3.8.so.1.0
# gcc sequence_gen_1.c -o sequence_gen_1.o -I$PYTHON_PATH -I$PYTHON_PATH/Include -L$PYTHON_PATH -L$PYTHON_PATH/libpython3.8.so.1.0 -L$PYTHON_PATH/libpython3.8.a
# gcc sequence_gen_1.c -o sequence_gen_1.o -I$PYTHON_PATH -I$PYTHON_PATH/Include -L$PYTHON_PATH/libpython3.8.so.1.0

#! /bin/bash

if [ "$#" -lt 1 ]; then
    suffix=10
    need_compile=false
    echo "suffix=${suffix}"
elif [ "$#" -lt 2 ]; then
    suffix=$1
    need_compile=false
elif [ "$#" -lt 3 ]; then
    suffix=$1
    need_compile=$2
fi

if [ need = true ]; then
    make
fi

time ./build/tetris_deep_search_path_optimization.o -a 5000 -f data_fields_${suffix} -u 3 -r 20 -c 10 -b 4
time python3 create_images_from_tetris_game_data.py ${suffix}

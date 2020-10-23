#! /bin/bash

usage() {
  echo "Usage: $0 [ -a ARG_A ] [ -f ARG_F ] [ -u ARG_U ] [ -r ARG_R ] [ -c ARG_C ] [ -b ARG_B ]" 1>&2
}

exit_abnormal() {
  usage
  exit 1
}

exit_abnormal_missing_argument() {
    echo "Missing an argument!"
    exit_abnormal
}

ARG_COMPILE=false
ARG_A=100
ARG_S=10
ARG_U=4
ARG_R=20
ARG_C=5
ARG_B=4


TEMP=`getopt -o :a:s:u:r:c:b: --long suffix:,amount-pieces:,compile -- "$@"`
eval set -- "$TEMP"
echo "TEMP: ${TEMP}"

while true ; do
    case "$1" in
        -a|--amount-pieces) ARG_A=$2; shift 2 ;;
        -s|--suffix) ARG_S=$2; shift 2 ;;
        -u) ARG_U=$2; shift 2 ;;
        -r) ARG_R=$2; shift 2 ;;
        -c) ARG_C=$2; shift 2 ;;
        -b) ARG_B=$2; shift 2 ;;
        --compile) ARG_COMPILE=true; shift 1 ;;
        :) ;;
        --) shift ; break ;;
        *) exit_abnormal ;;
    esac
done

if $ARG_COMPILE; then
    make
fi

echo "ARG_COMPILE"

time ./build/tetris_deep_search_path_optimization.o -a $ARG_A -f data_fields_${ARG_S} -u $ARG_U -r $ARG_R -c $ARG_C -b $ARG_B
time python3 create_images_from_tetris_game_data.py $ARG_S

# time ./build/tetris_deep_search_path_optimization.o -a 500 -f data_fields_${suffix} -u 3 -r 20 -c 10 -b 4
# time python3 create_images_from_tetris_game_data.py ${suffix}

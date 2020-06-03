#! /bin/bash

# CMD_PARAMS="./simple_example.sh -a 1000 -s 12_5_01 -r 12 -c 5 -b 4 -u 4"
# CMD_PARAMS="./simple_example.sh -a 1000 -s 12_5_1 -r 12 -c 5 -b 4 -u 4"

# ARG_A=1000
# ARG_R=12
# ARG_C=5
# ARG_B=3
# ARG_U=4

ARG_A=200
ARG_R=20
ARG_C=5
ARG_B=4
ARG_U=4

ARG_I_1=51
ARG_I_2=100

ARG_J_1=1
ARG_J_2=6

for i in `seq ${ARG_I_1} ${ARG_I_2}`; do
    echo "i: ${i}"
    for j in `seq ${ARG_J_1} ${ARG_J_2}`; do
        ./simple_example.sh -a ${ARG_A} -s r${ARG_R}_c${ARG_C}_i${i}_j${j}_b${ARG_B} -r ${ARG_R} -c ${ARG_C} -b ${ARG_B} -u ${ARG_U} > /dev/null 2>&1 &
    done
    wait
done

#! /bin/bash

for i in `seq 4 10`; do
    echo "i: ${i}"
    ./simple_example.sh 101_${i} > /dev/null 2>&1 &
    ./simple_example.sh 102_${i} > /dev/null 2>&1 &
    ./simple_example.sh 103_${i} > /dev/null 2>&1 &
    ./simple_example.sh 104_${i} > /dev/null 2>&1 &
    ./simple_example.sh 105_${i} > /dev/null 2>&1 &
    ./simple_example.sh 106_${i} > /dev/null 2>&1 &
    ./simple_example.sh 107_${i} > /dev/null 2>&1 &
    # ./simple_example.sh 4 > /dev/null 2>&1 &
    # ./simple_example.sh 5 > /dev/null 2>&1 &
    # ./simple_example.sh 6 > /dev/null 2>&1 &
    # ./simple_example.sh 7 > /dev/null 2>&1 &
    wait
done

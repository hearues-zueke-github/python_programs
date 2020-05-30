#! /bin/bash

for i in `seq 1 3`; do
    echo "i: ${i}"
    ./simple_example.sh 1 > /dev/null 2>&1 &
    ./simple_example.sh 2 > /dev/null 2>&1 &
    ./simple_example.sh 3 > /dev/null 2>&1 &
    # ./simple_example.sh 4 > /dev/null 2>&1 &
    # ./simple_example.sh 5 > /dev/null 2>&1 &
    # ./simple_example.sh 6 > /dev/null 2>&1 &
    # ./simple_example.sh 7 > /dev/null 2>&1 &
    wait
done

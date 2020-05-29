#! /bin/bash

# ./simple_example.sh 1 &
# ./simple_example.sh 2 &
# ./simple_example.sh 3 &
# ./simple_example.sh 4 &
# ./simple_example.sh 5 &
# ./simple_example.sh 6 &
# ./simple_example.sh 7 &
for i in `seq 1 10`; do
    echo "i: ${i}"
    ./simple_example.sh 1 &> /dev/null &
    ./simple_example.sh 2 &> /dev/null &
    ./simple_example.sh 3 &> /dev/null &
    ./simple_example.sh 4 &> /dev/null &
    ./simple_example.sh 5 &> /dev/null &
    ./simple_example.sh 6 &> /dev/null &
    ./simple_example.sh 7 &> /dev/null &
    wait
done

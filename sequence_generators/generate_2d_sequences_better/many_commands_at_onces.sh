#! /bin/bash

cd cpp/build
make -j 5 Generate2dSequence

rows=2048
cols=2048

modulo=3
for idx in {1..10}; do
	cmd="time ./Generate2dSequence $rows $cols 3 $modulo 0 1 1 0"
	echo "cmd: $cmd"
	$($cmd)
	modulo=$((modulo+1))
	# modulo=$((modulo+2))
done

# modulo=2
# cmd="time ./Generate2dSequence $rows $cols $modulo 1 1 1 1"
# echo "cmd: $cmd"
# $($cmd)

# modulo=$((modulo*2))
# cmd="time ./Generate2dSequence $rows $cols $modulo 1 1 1 1"
# echo "cmd: $cmd"
# $($cmd)

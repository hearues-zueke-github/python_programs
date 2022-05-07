#! /bin/bash

cd cpp/build
make -j 5 Generate2dSequence

rows=6048
cols=6048

modulo=7
for idx in {1..8}; do
	cmd="time ./Generate2dSequence $rows $cols $modulo 1 2 3 5"
	echo "cmd: $cmd"
	$($cmd)
	modulo=$((modulo+2))
done

# modulo=2
# cmd="time ./Generate2dSequence $rows $cols $modulo 1 1 1 1"
# echo "cmd: $cmd"
# $($cmd)

# modulo=$((modulo*2))
# cmd="time ./Generate2dSequence $rows $cols $modulo 1 1 1 1"
# echo "cmd: $cmd"
# $($cmd)

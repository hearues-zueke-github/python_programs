#! /bin/bash

read_string=$(./get_bash_array.py)
echo $read_string

IFS=';' read -ra my_array <<< "$read_string"

length=${#my_array[@]}
echo "Length of my_array is: ${length}"

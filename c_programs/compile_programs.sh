#! /bin/bash

# gcc -Wall test_currying.c -o test_currying
# echo "Compiled 'test_currying.c'"

gcc -Wall -pthread test_varibale_from_other_thread.c -o test_varibale_from_other_thread
echo "Compiled 'test_varibale_from_other_thread.c'"

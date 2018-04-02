#! /bin/bash

./encrypt_own_file.py en test_file.txt test_file_2.txt
./encrypt_own_file.py de test_file_2.txt test_file_3.txt

# The files test_file.txt and test_file_3.txt should be the same!
# No key needed for this right now!

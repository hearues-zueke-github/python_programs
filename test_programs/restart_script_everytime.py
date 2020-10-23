#! /usr/bin/python3

# -*- coding: utf-8 -*-

import os
import sys

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

if __name__=='__main__':
    # After each run of the script it will restart the whole script again,
    # and again, and again... after each press of the ENTER key!
    # You can change the code however you want until it fails at some point ;-)
    print('Hello World!224324')
    for i in range(0, 10):
        sys.stdout.write(f'{i} ')
        sys.stdout.flush()
    input('Press ENTER to restart...')
    os.execv(sys.executable, ['python3'] + sys.argv)
